# python import is executed before modal ships the code
# the local file config is in the parent directory thus missing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal
from typing import Optional

# Modal setup
app = modal.App("agd-preprocess-charts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "boto3",
        "psycopg2-binary",
        "Pillow",
        "numpy",
    )
    .add_local_file("../aws.py", "/root/aws.py")
    .add_local_file("../config.py", "/root/config.py")
)

# Configuration
TARGET_SIZE = 512
PAD_COLOR = (255, 255, 255)
WHITESPACE_CROP_THRESHOLD = 245
WHITESPACE_CROP_MIN_BORDER = 5
BATCH_SIZE = 100



# Preprocessing functions
def validate_image(img_bytes: bytes) -> bool:
    from PIL import Image
    import io

    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.verify()
        return True
    except Exception:
        return False


def convert_to_rgb(img):
    from PIL import Image
    if img.mode == "RGB":
        return img
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, PAD_COLOR)
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode == "P":
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, PAD_COLOR)
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode == "LA":
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, PAD_COLOR)
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def auto_crop_whitespace(img):
    import numpy as np
    arr = np.array(img)
    mask = np.any(arr < WHITESPACE_CROP_THRESHOLD, axis=2)
    if not mask.any():
        return img, [0, 0, img.width, img.height]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    margin = WHITESPACE_CROP_MIN_BORDER
    rmin = max(0, rmin - margin)
    rmax = min(img.height - 1, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(img.width - 1, cmax + margin)
    crop_box = [cmin, rmin, cmax + 1, rmax + 1]
    return img.crop(crop_box), crop_box


def letterbox_resize(img, target_size: int = TARGET_SIZE):
    from PIL import Image
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), PAD_COLOR)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas, {
        "scale": round(scale, 6),
        "offset_x": offset_x,
        "offset_y": offset_y,
        "resized_w": new_w,
        "resized_h": new_h,
    }


def preprocess_single(img_bytes: bytes) -> Optional[dict]:
    from PIL import Image
    import io

    if not validate_image(img_bytes):
        return None

    img = Image.open(io.BytesIO(img_bytes))
    img = convert_to_rgb(img)
    original_width, original_height = img.size
    img, crop_box = auto_crop_whitespace(img)
    img, letterbox_meta = letterbox_resize(img, TARGET_SIZE)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)

    return {
        "image_bytes": buf.getvalue(),
        "original_width": original_width,
        "original_height": original_height,
        "meta": {
            "crop_box": crop_box,
            "scale": letterbox_meta["scale"],
            "offset_x": letterbox_meta["offset_x"],
            "offset_y": letterbox_meta["offset_y"],
            "resized_w": letterbox_meta["resized_w"],
            "resized_h": letterbox_meta["resized_h"],
            "target_size": TARGET_SIZE,
        },
    }


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("aws"),
        modal.Secret.from_name("aws-rds"),
    ],
    timeout=7200,
    memory=4096,
)
def preprocess_all():
    import aws
    import json
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("preprocess_charts")

    # Get unique raw images needing processing (any dataset)
    with aws.get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT raw_graph
                FROM samples
                WHERE good_graph IS NULL
            """)
            raw_uuids = [row[0] for row in cur.fetchall()]

    log.info(f"Found {len(raw_uuids)} unique images to preprocess")

    if not raw_uuids:
        log.info("Nothing to process")
        return {"unique_images": 0, "rows_updated": 0, "skipped": 0}

    # Process each unique image
    processed_count = 0
    skipped_count = 0
    rows_updated = 0

    with aws.get_db_connection() as conn:
        with conn.cursor() as cur:
            for raw_uuid in raw_uuids:
                # Download via get_image
                try:
                    raw_bytes = aws.get_image(raw_uuid)
                except KeyError:
                    log.warning(f"  Not found in S3: {raw_uuid}")
                    skipped_count += 1
                    continue

                # Preprocess
                result = preprocess_single(raw_bytes)
                if result is None:
                    log.warning(f"  Corrupt image: {raw_uuid}")
                    skipped_count += 1
                    continue

                # Upload processed image → new UUID
                processed_uuid = aws.put_image(result["image_bytes"])

                # Update all rows sharing this raw_graph
                meta_json = json.dumps(result["meta"])
                original_width = result["original_width"]
                original_height = result["original_height"]
                cur.execute(
                    """
                    UPDATE samples
                    SET good_graph = %s, preprocess_meta = %s, original_width = %s, original_height = %s
                    WHERE raw_graph = %s AND good_graph IS NULL
                    """,
                    (str(processed_uuid), meta_json, original_width, original_height, str(raw_uuid)),
                )
                rows_updated += cur.rowcount
                processed_count += 1

                if processed_count % BATCH_SIZE == 0:
                    conn.commit()
                    log.info(
                        f"  Processed {processed_count}/{len(raw_uuids)} images "
                        f"({rows_updated} rows updated)"
                    )

    log.info(f"Done: {processed_count} images, "
             f"{rows_updated} rows updated, {skipped_count} skipped")

    return {
        "unique_images": processed_count,
        "rows_updated": rows_updated,
        "skipped": skipped_count,
    }

# Entrypoint
@app.local_entrypoint()
def main():
    result = preprocess_all.remote()
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
