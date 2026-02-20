"""
Modal preprocessor: Prepare chart images for AGD attack

Reads raw images from S3 (good_graphs/), applies preprocessing, writes
results to S3 (processed/), and updates RDS with processed paths + metadata.

Preprocessing steps:
  1. Validate image integrity (detect corrupt files)
  2. Convert to RGB (handle RGBA, grayscale, palette modes)
  3. Auto-crop excess whitespace borders (maximize chart area in frame)
  4. Letterbox pad to square (preserve aspect ratio — critical for charts)
  5. Resize to 512×512 with LANCZOS antialiasing
  6. Normalize to consistent white background (replace transparency)
  7. Record original dimensions + preprocessing metadata in RDS

Why each step matters for AGD:
  - 512×512 RGB is the native input for Stable Diffusion 1.5's VAE encoder
  - Letterbox padding preserves bar widths, axis proportions, line slopes —
    stretching would distort these and the adversarial perturbation wouldn't
    transfer back to the original geometry
  - Whitespace cropping maximizes the chart's footprint within 512×512,
    giving the diffusion model more chart signal vs. blank padding
  - RGB conversion prevents channel mismatch errors in the VAE
  - Consistent white background avoids artifacts from transparency
"""

import modal
import os
import io
import logging
from typing import Optional

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("agd-preprocess-charts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "boto3",
        "psycopg2-binary",
        "Pillow",
        "numpy",
    )
)

# AWS and HuggingFaceConfiguration
S3_BUCKET = "agd-dev-tyson"
S3_RAW_PREFIX = "good_graphs/ChartBench"
S3_PROCESSED_PREFIX = "processed/ChartBench"
AWS_REGION = "ca-central-1"

# Define preprocessing parameters
TARGET_SIZE = 512
PAD_COLOR = (255, 255, 255)
WHITESPACE_CROP_THRESHOLD = 245
WHITESPACE_CROP_MIN_BORDER = 5
BATCH_SIZE = 100

# Preprocessing functions

def validate_image(img_bytes: bytes) -> bool:
    """Check that the image bytes decode without error."""
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.verify()
        return True
    except Exception:
        return False


def convert_to_rgb(img):
    """
    Force-convert any image mode to RGB with white background.

    Handles:
      - RGBA → composite onto white background (common for chart PNGs)
      - P (palette) → convert via RGBA to handle palette transparency
      - L (grayscale) → direct RGB conversion
      - LA (grayscale + alpha) → composite onto white
    """
    from PIL import Image

    if img.mode == "RGB":
        return img

    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, PAD_COLOR)
        background.paste(img, mask=img.split()[3])  # alpha channel as mask
        return background

    if img.mode == "P":
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, PAD_COLOR)
        background.paste(img, mask=img.split()[3])
        return background

    if img.mode == "LA":
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, PAD_COLOR)
        background.paste(img, mask=img.split()[3])
        return background

    # L (grayscale), CMYK, etc.
    return img.convert("RGB")


def auto_crop_whitespace(img):
    """
    Remove excess whitespace borders around the chart content.

    Uses numpy to find the bounding box of non-white pixels, then crops
    with a small margin. This maximizes chart area within the final 512×512
    frame, giving the diffusion model more signal to work with.

    Returns the cropped image and the crop box (for metadata).
    """
    import numpy as np

    arr = np.array(img)

    # Find non-white pixels (any channel below threshold)
    mask = np.any(arr < WHITESPACE_CROP_THRESHOLD, axis=2)

    if not mask.any():
        # Entirely white/blank image — return as-is
        return img, (0, 0, img.width, img.height)

    # Bounding box of non-white content
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add small border margin (but don't exceed image bounds)
    margin = WHITESPACE_CROP_MIN_BORDER
    rmin = max(0, rmin - margin)
    rmax = min(img.height - 1, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(img.width - 1, cmax + margin)

    crop_box = (cmin, rmin, cmax + 1, rmax + 1)
    return img.crop(crop_box), crop_box


def letterbox_resize(img, target_size: int = TARGET_SIZE):
    """
    Resize image to target_size × target_size with letterbox padding.

    Instead of stretching (which distorts bar widths, axis proportions,
    and line slopes), we:
      1. Scale the image so its longest side = target_size
      2. Pad the shorter side with white to make it square

    This preserves the chart's geometric properties, which matters because
    AGD's adversarial perturbation operates in this 512×512 space — if the
    geometry is distorted, the perturbation won't transfer meaningfully.
    """
    from PIL import Image

    w, h = img.size
    scale = target_size / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    # Resize with high-quality LANCZOS resampling
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create white square canvas and paste centered
    canvas = Image.new("RGB", (target_size, target_size), PAD_COLOR)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas, {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "resized_w": new_w,
        "resized_h": new_h,
    }


def preprocess_single(img_bytes: bytes) -> Optional[dict]:
    """
    Full preprocessing pipeline for one image.

    Returns dict with:
      - "image_bytes": processed PNG bytes ready for S3 upload
      - "original_w", "original_h": original dimensions
      - "crop_box": whitespace crop applied
      - "letterbox": padding/scale metadata (for inverse transform later)
    Or None if the image is corrupt.
    """
    from PIL import Image

    if not validate_image(img_bytes):
        return None

    img = Image.open(io.BytesIO(img_bytes))
    original_w, original_h = img.size
    original_mode = img.mode

    # Step 1: Convert to RGB
    img = convert_to_rgb(img)

    # Step 2: Auto-crop whitespace
    img, crop_box = auto_crop_whitespace(img)

    # Step 3: Letterbox resize to 512×512
    img, letterbox_meta = letterbox_resize(img, TARGET_SIZE)

    # Step 4: Encode as PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    processed_bytes = buf.getvalue()

    return {
        "image_bytes": processed_bytes,
        "original_w": original_w,
        "original_h": original_h,
        "original_mode": original_mode,
        "crop_box": crop_box,
        "letterbox": letterbox_meta,
    }


# ---------------------------------------------------------------------------
# S3 key mapping: raw → processed
# ---------------------------------------------------------------------------

def raw_key_to_processed_key(raw_key: str) -> str:
    """
    Map a raw S3 key to its processed counterpart.

    Example:
      "good_graphs/ChartBench/test/area/area/chart_0.png"
      → "processed/ChartBench/test/area/area/chart_0.png"
    """
    return raw_key.replace(S3_RAW_PREFIX, S3_PROCESSED_PREFIX, 1)


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("aws"),
        modal.Secret.from_name("aws-rds"),
    ],
    timeout=7200,    # 2 hours — image processing takes longer
    memory=4096,     # 4 GB — numpy arrays for whitespace detection
)
def preprocess_all():
    """Read raw images from S3, preprocess, write back, update RDS."""
    import boto3
    import psycopg2
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("preprocess_charts")

    # ── Connect to S3 ────────────────────────────────────────────────────
    s3 = boto3.client("s3", region_name=AWS_REGION)
    log.info(f"Connected to S3: {S3_BUCKET}")

    # ── Connect to RDS ───────────────────────────────────────────────────
    conn = psycopg2.connect(
        host="agd-dev-postgres.cdsyi46ammw7.ca-central-1.rds.amazonaws.com",
        port=5432,
        database="postgres",
        user="postgres",
        password=os.environ["DB_PASSWORD"],
        sslmode="require",
    )
    conn.autocommit = False
    cur = conn.cursor()
    log.info("Connected to RDS")

    # ── Add preprocessing columns if they don't exist ────────────────────
    for col_def in [
        "processed_graph TEXT",
        "original_w INTEGER",
        "original_h INTEGER",
        "preprocess_meta JSONB",
    ]:
        col_name = col_def.split()[0]
        cur.execute(f"""
            DO $$
            BEGIN
                ALTER TABLE samples ADD COLUMN {col_def};
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END $$;
        """)
    conn.commit()
    log.info("Schema updated with preprocessing columns")

    # ── Query all ChartBench samples that haven't been processed yet ─────
    cur.execute("""
        SELECT id, good_graph
        FROM samples
        WHERE source = 'ChartBench'
          AND processed_graph IS NULL
        ORDER BY id
    """)
    rows = cur.fetchall()
    total = len(rows)
    log.info(f"Found {total} unprocessed samples")

    if total == 0:
        log.info("Nothing to process")
        cur.close()
        conn.close()
        return {"total": 0, "processed": 0, "skipped": 0}

    # ── Process each image ───────────────────────────────────────────────
    UPDATE_SQL = """
        UPDATE samples
        SET processed_graph = %s,
            original_w = %s,
            original_h = %s,
            preprocess_meta = %s
        WHERE id = %s
    """

    processed_count = 0
    skipped_count = 0
    s3_cache: dict[str, str] = {}  # raw_key → processed_key (dedup uploads)

    for i, (sample_id, good_graph_uri) in enumerate(rows):
        # Parse S3 URI: "s3://adviz-charts/good_graphs/ChartBench/..."
        raw_key = good_graph_uri.replace(f"s3://{S3_BUCKET}/", "")
        processed_key = raw_key_to_processed_key(raw_key)
        processed_uri = f"s3://{S3_BUCKET}/{processed_key}"

        # ── Download raw image (skip if already processed this key) ──
        if raw_key not in s3_cache:
            try:
                resp = s3.get_object(Bucket=S3_BUCKET, Key=raw_key)
                raw_bytes = resp["Body"].read()
            except Exception as e:
                log.warning(f"  Failed to download {raw_key}: {e}")
                skipped_count += 1
                continue

            # ── Preprocess ───────────────────────────────────────────
            result = preprocess_single(raw_bytes)
            if result is None:
                log.warning(f"  Corrupt image: {raw_key}")
                skipped_count += 1
                continue

            # ── Upload processed image to S3 ─────────────────────────
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=processed_key,
                Body=result["image_bytes"],
                ContentType="image/png",
            )

            # Cache for dedup (multiple QA rows share the same image)
            s3_cache[raw_key] = json.dumps({
                "original_w": result["original_w"],
                "original_h": result["original_h"],
                "original_mode": result["original_mode"],
                "crop_box": list(result["crop_box"]),
                "letterbox": result["letterbox"],
                "target_size": TARGET_SIZE,
            })

        # ── Update RDS row ───────────────────────────────────────────
        meta_json = s3_cache[raw_key]
        meta = json.loads(meta_json)

        cur.execute(UPDATE_SQL, (
            processed_uri,
            meta["original_w"],
            meta["original_h"],
            meta_json,
            sample_id,
        ))
        processed_count += 1

        if processed_count % BATCH_SIZE == 0:
            conn.commit()
            log.info(
                f"  Processed {processed_count}/{total} rows "
                f"({len(s3_cache)} unique images, {skipped_count} skipped)"
            )

    # Final commit
    conn.commit()
    log.info(
        f"Done: {processed_count} processed, {skipped_count} skipped, "
        f"{len(s3_cache)} unique images"
    )

    cur.close()
    conn.close()

    return {
        "total": total,
        "processed": processed_count,
        "skipped": skipped_count,
        "unique_images": len(s3_cache),
    }


# ---------------------------------------------------------------------------
# Entrypoint — run with: modal run preprocess_charts.py
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    result = preprocess_all.remote()
    print("\n" + "=" * 60)
    print("Chart Preprocessing Summary")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60)