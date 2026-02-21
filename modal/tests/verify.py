import modal
import io
import logging

from aws import get_db_connection, get_image

app = modal.App("agd-verify")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boto3", "psycopg2-binary", "Pillow")
    .add_local_file("aws.py", "/root/aws.py")
    .add_local_file("config.py", "/root/config.py")
)

TARGET_SIZE = 512


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("aws")],
    timeout=600, memory=2048,
)
def verify(count: int = 10, check_all: bool = False):
    from PIL import Image

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("verify")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Summary stats
            cur.execute("SELECT COUNT(*) FROM samples")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM samples WHERE good_graph IS NOT NULL")
            preprocessed = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT raw_graph) FROM samples")
            unique_raw = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT good_graph) FROM samples WHERE good_graph IS NOT NULL")
            unique_proc = cur.fetchone()[0]

            log.info(f"RDS summary:")
            log.info(f"  Total rows:        {total}")
            log.info(f"  Preprocessed:      {preprocessed}")
            log.info(f"  Unique raw images: {unique_raw}")
            log.info(f"  Unique processed:  {unique_proc}")

            if total == 0:
                log.warning("No rows in samples table")
                return {"total": 0, "checked": 0, "errors": 0}

            # Fetch samples to check
            limit = "" if check_all else f"LIMIT {count}"
            cur.execute(f"""
                SELECT id, source, graph_type, raw_graph, good_graph,
                       original_width, original_height
                FROM samples
                ORDER BY id
                {limit}
            """)
            rows = cur.fetchall()

    log.info(f"\nChecking {len(rows)} samples...\n")

    errors = []
    raw_checked = set()
    proc_checked = set()

    for sid, source, gtype, raw_uuid, proc_uuid, ow, oh in rows:
        # Check raw image
        if raw_uuid not in raw_checked:
            try:
                raw_bytes = get_image(raw_uuid)
                img = Image.open(io.BytesIO(raw_bytes))
                img.verify()
                raw_checked.add(raw_uuid)
            except KeyError:
                errors.append(f"#{sid}: raw_graph {raw_uuid} missing from S3")
                continue
            except Exception as e:
                errors.append(f"#{sid}: raw_graph {raw_uuid} corrupt — {e}")
                continue

        # Check processed image
        if proc_uuid and proc_uuid not in proc_checked:
            try:
                proc_bytes = get_image(proc_uuid)
                img = Image.open(io.BytesIO(proc_bytes))
                w, h = img.size
                mode = img.mode
                if (w, h) != (TARGET_SIZE, TARGET_SIZE):
                    errors.append(f"#{sid}: good_graph {proc_uuid} is {w}x{h}, expected {TARGET_SIZE}x{TARGET_SIZE}")
                elif mode != "RGB":
                    errors.append(f"#{sid}: good_graph {proc_uuid} is {mode}, expected RGB")
                else:
                    proc_checked.add(proc_uuid)
            except KeyError:
                errors.append(f"#{sid}: good_graph {proc_uuid} missing from S3")
            except Exception as e:
                errors.append(f"#{sid}: good_graph {proc_uuid} corrupt — {e}")

    checked = len(rows)
    log.info(f"Checked {checked} rows ({len(raw_checked)} raw, {len(proc_checked)} processed)")

    if errors:
        log.error(f"\n{len(errors)} ERRORS:")
        for e in errors:
            log.error(f"{e}")
    else:
        log.info("All checks passed ✓")

    return {
        "total_rows": total,
        "checked": checked,
        "raw_ok": len(raw_checked),
        "processed_ok": len(proc_checked),
        "errors": len(errors),
        "error_details": errors[:20],
    }


@app.local_entrypoint()
def main(count: int = 10, all: bool = False):
    r = verify.remote(count=count, check_all=all)
    print(f"\n{'='*40}")
    print(f"  total_rows:   {r['total_rows']}")
    print(f"  checked:      {r['checked']}")
    print(f"  raw_ok:       {r['raw_ok']}")
    print(f"  processed_ok: {r['processed_ok']}")
    print(f"  errors:       {r['errors']}")
    if r["error_details"]:
        print("\n  Error details:")
        for e in r["error_details"]:
            print(f"    ✗ {e}")
    print(f"\n{'all checks passed' if r['errors'] == 0 else 'some failed'}")
