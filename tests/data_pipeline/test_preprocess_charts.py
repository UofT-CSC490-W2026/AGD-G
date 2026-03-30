import io
import sys
import types
from unittest.mock import MagicMock

from PIL import Image


def make_png(mode="RGB", size=(32, 16), color=None):
    image = Image.new(mode, size, color or ((255, 0, 0) if mode == "RGB" else (255, 0, 0, 128)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _install_fake_boto(monkeypatch):
    """Mock boto3/psycopg2/botocore so importing aws doesn't need real AWS."""
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = MagicMock()
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_psycopg2.connect = MagicMock()
    fake_botocore = types.ModuleType("botocore")
    fake_botocore_exc = types.ModuleType("botocore.exceptions")
    fake_botocore_exc.ClientError = type("ClientError", (Exception,), {})
    fake_botocore.exceptions = fake_botocore_exc

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_botocore_exc)


def _load_module(monkeypatch):
    import importlib
    _install_fake_boto(monkeypatch)
    sys.modules.pop("agdg.data_pipeline.aws", None)
    sys.modules.pop("agdg.data_pipeline.preprocess_charts", None)
    return importlib.import_module("agdg.data_pipeline.preprocess_charts")


# ---------------------------------------------------------------------------
# Pure function tests (no aws interaction)
# ---------------------------------------------------------------------------

def test_validate_image_accepts_png_and_rejects_garbage(monkeypatch):
    module = _load_module(monkeypatch)
    assert module.validate_image(make_png()) is True
    assert module.validate_image(b"not-an-image") is False


def test_convert_to_rgb_flattens_alpha(monkeypatch):
    module = _load_module(monkeypatch)
    image = Image.new("RGBA", (10, 10), (0, 128, 255, 128))
    converted = module.convert_to_rgb(image)
    assert converted.mode == "RGB"
    assert converted.size == (10, 10)


def test_convert_to_rgb_handles_palette_and_la_and_other_modes(monkeypatch):
    module = _load_module(monkeypatch)
    assert module.convert_to_rgb(Image.new("P", (4, 4))).mode == "RGB"
    assert module.convert_to_rgb(Image.new("LA", (4, 4), (100, 128))).mode == "RGB"
    assert module.convert_to_rgb(Image.new("L", (4, 4), 50)).mode == "RGB"


def test_auto_crop_whitespace_returns_tighter_box(monkeypatch):
    module = _load_module(monkeypatch)
    image = Image.new("RGB", (30, 30), (255, 255, 255))
    for x in range(10, 15):
        for y in range(12, 18):
            image.putpixel((x, y), (0, 0, 0))

    cropped, crop_box = module.auto_crop_whitespace(image)
    assert cropped.size[0] < image.size[0]
    assert cropped.size[1] < image.size[1]
    assert crop_box[0] <= 10
    assert crop_box[1] <= 12


def test_auto_crop_whitespace_returns_original_for_blank_image(monkeypatch):
    module = _load_module(monkeypatch)
    image = Image.new("RGB", (20, 20), (255, 255, 255))
    cropped, crop_box = module.auto_crop_whitespace(image)
    assert cropped is image
    assert crop_box == [0, 0, 20, 20]


def test_letterbox_resize_outputs_target_square(monkeypatch):
    module = _load_module(monkeypatch)
    image = Image.new("RGB", (200, 100), (255, 255, 255))
    resized, meta = module.letterbox_resize(image, target_size=128)
    assert resized.size == (128, 128)
    assert meta["resized_w"] == 128
    assert meta["resized_h"] == 64
    assert meta["offset_y"] == 32


def test_preprocess_single_returns_png_and_metadata(monkeypatch):
    module = _load_module(monkeypatch)
    result = module.preprocess_single(make_png(size=(60, 20)))
    assert result is not None
    assert result["original_width"] == 60
    assert result["original_height"] == 20
    assert result["meta"]["target_size"] == module.TARGET_SIZE
    processed = Image.open(io.BytesIO(result["image_bytes"]))
    assert processed.size == (module.TARGET_SIZE, module.TARGET_SIZE)


def test_preprocess_single_returns_none_for_invalid_image(monkeypatch):
    module = _load_module(monkeypatch)
    assert module.preprocess_single(b"not-an-image") is None


# ---------------------------------------------------------------------------
# preprocess_all tests (mock aws)
# ---------------------------------------------------------------------------

def _build_fake_aws(raw_uuids, updates, *, missing_keys=None):
    missing_keys = missing_keys or set()

    class FakeCursor:
        def __init__(self):
            self.rowcount = 2

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConnection:
        def __init__(self):
            self.commits = 0
            self.cursor_obj = FakeCursor()

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.commits += 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConnectionFactory:
        def __init__(self):
            self.connections = []

        def __call__(self):
            conn = FakeConnection()
            self.connections.append(conn)
            return conn

    factory = FakeConnectionFactory()

    fake_aws = types.ModuleType("agdg.data_pipeline.aws")
    fake_aws.rds = types.SimpleNamespace(
        get_db_connection=factory,
        iter_preprocessor_inputs=lambda conn: [
            {"sample_id": idx + 1, "raw_chart": raw_uuid}
            for idx, raw_uuid in enumerate(raw_uuids)
        ],
        insert_preprocessing=lambda cur, sample_id, chart, original_width, original_height, meta: updates.append(
            (str(chart), sample_id, original_width, original_height, meta)
        ),
    )

    def get_image(key):
        if key in missing_keys:
            raise KeyError(key)
        return b"raw-image"

    fake_aws.s3 = types.SimpleNamespace(
        get_image=get_image,
        put_image=lambda image: "processed-uuid",
    )
    return fake_aws, factory


def test_preprocess_all_updates_rows_with_processed_images(monkeypatch):
    module = _load_module(monkeypatch)
    updates = []
    fake_aws, _ = _build_fake_aws(["raw-1"], updates)
    monkeypatch.setattr(module, "rds", fake_aws.rds)
    monkeypatch.setattr(module, "s3", fake_aws.s3)
    monkeypatch.setattr(
        module,
        "preprocess_single",
        lambda img_bytes: {
            "image_bytes": b"processed-image",
            "original_width": 60,
            "original_height": 20,
            "meta": {"crop_box": [0, 0, 10, 10]},
        },
    )

    result = module.preprocess_all()

    assert result == {"unique_images": 1, "rows_updated": 2, "skipped": 0}
    assert updates[0][0] == "processed-uuid"
    assert updates[0][2:4] == (60, 20)


def test_preprocess_all_returns_zero_when_nothing_to_process(monkeypatch):
    module = _load_module(monkeypatch)
    updates = []
    fake_aws, _ = _build_fake_aws([], updates)
    monkeypatch.setattr(module, "rds", fake_aws.rds)
    monkeypatch.setattr(module, "s3", fake_aws.s3)

    result = module.preprocess_all()

    assert result == {"unique_images": 0, "rows_updated": 0, "skipped": 0}
    assert updates == []


def test_preprocess_all_skips_missing_and_corrupt_images(monkeypatch):
    module = _load_module(monkeypatch)
    updates = []
    fake_aws, _ = _build_fake_aws(["missing", "corrupt"], updates, missing_keys={"missing"})
    monkeypatch.setattr(module, "rds", fake_aws.rds)
    monkeypatch.setattr(module, "s3", fake_aws.s3)
    monkeypatch.setattr(module, "preprocess_single", lambda img_bytes: None)

    result = module.preprocess_all()

    assert result == {"unique_images": 0, "rows_updated": 0, "skipped": 2}
    assert updates == []


def test_preprocess_all_commits_batch_updates(monkeypatch):
    module = _load_module(monkeypatch)
    module.BATCH_SIZE = 1
    updates = []
    fake_aws, factory = _build_fake_aws(["raw-1"], updates)
    monkeypatch.setattr(module, "rds", fake_aws.rds)
    monkeypatch.setattr(module, "s3", fake_aws.s3)
    monkeypatch.setattr(
        module,
        "preprocess_single",
        lambda img_bytes: {
            "image_bytes": b"processed-image",
            "original_width": 60,
            "original_height": 20,
            "meta": {"crop_box": [0, 0, 10, 10]},
        },
    )

    result = module.preprocess_all()

    assert result["unique_images"] == 1
    assert factory.connections[0].commits == 1
