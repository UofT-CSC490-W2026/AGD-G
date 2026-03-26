import io
import sys

from PIL import Image

from conftest import ensure_modal_root, import_fresh, install_fake_modal


def make_png(mode="RGB", size=(32, 16), color=None):
    image = Image.new(mode, size, color or ((255, 0, 0) if mode == "RGB" else (255, 0, 0, 128)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_module():
    ensure_modal_root()
    install_fake_modal()
    return import_fresh("data_pipeline.preprocess_charts")


def test_validate_image_accepts_png_and_rejects_garbage():
    module = load_module()

    assert module.validate_image(make_png()) is True
    assert module.validate_image(b"not-an-image") is False


def test_convert_to_rgb_flattens_alpha():
    module = load_module()
    image = Image.new("RGBA", (10, 10), (0, 128, 255, 128))

    converted = module.convert_to_rgb(image)

    assert converted.mode == "RGB"
    assert converted.size == (10, 10)


def test_convert_to_rgb_handles_palette_and_la_and_other_modes():
    module = load_module()

    palette = Image.new("P", (4, 4))
    converted_palette = module.convert_to_rgb(palette)
    converted_la = module.convert_to_rgb(Image.new("LA", (4, 4), (100, 128)))
    converted_l = module.convert_to_rgb(Image.new("L", (4, 4), 50))

    assert converted_palette.mode == "RGB"
    assert converted_la.mode == "RGB"
    assert converted_l.mode == "RGB"


def test_auto_crop_whitespace_returns_tighter_box():
    module = load_module()
    image = Image.new("RGB", (30, 30), (255, 255, 255))
    for x in range(10, 15):
        for y in range(12, 18):
            image.putpixel((x, y), (0, 0, 0))

    cropped, crop_box = module.auto_crop_whitespace(image)

    assert cropped.size[0] < image.size[0]
    assert cropped.size[1] < image.size[1]
    assert crop_box[0] <= 10
    assert crop_box[1] <= 12


def test_auto_crop_whitespace_returns_original_for_blank_image():
    module = load_module()
    image = Image.new("RGB", (20, 20), (255, 255, 255))

    cropped, crop_box = module.auto_crop_whitespace(image)

    assert cropped is image
    assert crop_box == [0, 0, 20, 20]


def test_letterbox_resize_outputs_target_square():
    module = load_module()
    image = Image.new("RGB", (200, 100), (255, 255, 255))

    resized, meta = module.letterbox_resize(image, target_size=128)

    assert resized.size == (128, 128)
    assert meta["resized_w"] == 128
    assert meta["resized_h"] == 64
    assert meta["offset_y"] == 32


def test_preprocess_single_returns_png_and_metadata():
    module = load_module()

    result = module.preprocess_single(make_png(size=(60, 20)))

    assert result is not None
    assert result["original_width"] == 60
    assert result["original_height"] == 20
    assert result["meta"]["target_size"] == module.TARGET_SIZE
    processed = Image.open(io.BytesIO(result["image_bytes"]))
    assert processed.size == (module.TARGET_SIZE, module.TARGET_SIZE)


def test_preprocess_single_returns_none_for_invalid_image():
    module = load_module()

    assert module.preprocess_single(b"not-an-image") is None


def build_fake_aws(raw_uuids, updates, *, missing_keys=None, preprocess_result=None):
    missing_keys = missing_keys or set()

    class FakeCursor:
        def __init__(self):
            self.rowcount = 2

        def execute(self, query, params=None):
            if query.lstrip().startswith("UPDATE"):
                updates.append(params)

        def fetchall(self):
            return [(raw_uuid,) for raw_uuid in raw_uuids]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConnection:
        def __init__(self):
            self.commits = 0

        def cursor(self):
            return FakeCursor()

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
    fake_aws = type("FakeAws", (), {})()
    fake_aws.get_db_connection = factory

    def get_image(key):
        if key in missing_keys:
            raise KeyError(key)
        return b"raw-image"

    fake_aws.get_image = get_image
    fake_aws.put_image = lambda image: "processed-uuid"
    return fake_aws, factory


def test_preprocess_all_updates_rows_with_processed_images(monkeypatch):
    module = load_module()
    updates = []
    fake_aws, _ = build_fake_aws(["raw-1"], updates)

    monkeypatch.setitem(sys.modules, "aws", fake_aws)
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
    module = load_module()
    updates = []
    fake_aws, _ = build_fake_aws([], updates)
    monkeypatch.setitem(sys.modules, "aws", fake_aws)

    result = module.preprocess_all()

    assert result == {"unique_images": 0, "rows_updated": 0, "skipped": 0}
    assert updates == []


def test_preprocess_all_skips_missing_and_corrupt_images(monkeypatch):
    module = load_module()
    updates = []
    fake_aws, _ = build_fake_aws(["missing", "corrupt"], updates, missing_keys={"missing"})
    monkeypatch.setitem(sys.modules, "aws", fake_aws)
    monkeypatch.setattr(
        module,
        "preprocess_single",
        lambda img_bytes: None,
    )

    result = module.preprocess_all()

    assert result == {"unique_images": 0, "rows_updated": 0, "skipped": 2}
    assert updates == []


def test_preprocess_all_commits_batch_updates(monkeypatch):
    module = load_module()
    module.BATCH_SIZE = 1
    updates = []
    fake_aws, factory = build_fake_aws(["raw-1"], updates)
    monkeypatch.setitem(sys.modules, "aws", fake_aws)
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
    assert factory.connections[1].commits == 1


def test_preprocess_main_prints_remote_result(capsys):
    module = load_module()
    module.preprocess_all.remote = lambda: {"unique_images": 2, "rows_updated": 3, "skipped": 1}

    module.main()

    out = capsys.readouterr().out
    assert "unique_images" in out
    assert "rows_updated" in out
