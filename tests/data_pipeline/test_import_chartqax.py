"""Tests for agdg.data_pipeline.import_chartqax."""
import importlib
import io
import os
import sys
import types

import pytest
from PIL import Image

from .conftest import build_fake_aws, make_png


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def _load(monkeypatch, fake_boto):
    for mod in list(sys.modules):
        if mod.startswith("agdg.data_pipeline.") and mod != "agdg.data_pipeline":
            sys.modules.pop(mod, None)
    return importlib.import_module("agdg.data_pipeline.import_chartqax")


def _install_fake_aws(module, fake_aws, monkeypatch):
    monkeypatch.setattr(module, "rds", fake_aws.rds)
    monkeypatch.setattr(module, "s3", fake_aws.s3)


# ---------------------------------------------------------------------------
# HuggingFace / filesystem fakes
# ---------------------------------------------------------------------------

_real_image_open = Image.open


def _install_fake_hf(monkeypatch, splits, *, zip_filename=""):
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *a, **kw: splits
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda **kw: "/fake/chartqa-x.zip"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    monkeypatch.setenv("HF_IMAGE_ZIP", zip_filename)
    monkeypatch.setenv("HF_IMAGE_ROOT", "ChartQA-X")


def _install_fake_fs(monkeypatch):
    class _FakeZipFile:
        def __init__(self, *a, **kw):
            pass
        def extractall(self, path):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr("zipfile.ZipFile", _FakeZipFile)
    monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)

    def fake_image_open(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and "ChartQA-X" in str(path):
            return Image.new("RGB", (32, 32), (255, 0, 0))
        return _real_image_open(path, *args, **kwargs)

    monkeypatch.setattr("PIL.Image.open", fake_image_open)


# ---------------------------------------------------------------------------
# _map_graph_type (pure)
# ---------------------------------------------------------------------------

def test_map_graph_type_known(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    from agdg.data_pipeline.chart_type import ChartType

    assert module._map_graph_type("two_col") == ChartType.BAR
    assert module._map_graph_type("multi_col") == ChartType.BAR


def test_map_graph_type_unknown(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    from agdg.data_pipeline.chart_type import ChartType

    assert module._map_graph_type("alien") == ChartType.OTHER


# ---------------------------------------------------------------------------
# _image_to_bytes
# ---------------------------------------------------------------------------

def test_image_to_bytes_returns_valid_png(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    img = Image.new("RGB", (10, 10), (0, 128, 255))

    data = module._image_to_bytes(img)

    assert isinstance(data, bytes)
    assert len(data) > 0
    restored = Image.open(io.BytesIO(data))
    assert restored.format == "PNG"
    assert restored.size == (10, 10)


# ---------------------------------------------------------------------------
# _get_image
# ---------------------------------------------------------------------------

def test_get_image_from_dict_with_bytes(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    img_bytes = make_png(size=(8, 8))
    row = {"image": {"bytes": img_bytes, "path": None}}

    result = module._get_image(row, "/tmp/extract", "ChartQA-X")

    assert isinstance(result, Image.Image)
    assert result.size == (8, 8)


def test_get_image_from_pil_object(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    pil_img = Image.new("RGB", (12, 12), (0, 0, 255))
    row = {"image": pil_img}

    result = module._get_image(row, "/tmp/extract", "ChartQA-X")

    assert result is pil_img


def test_get_image_from_dict_with_path(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    row = {"image": {"bytes": None, "path": "/some/local/image.png"}}

    monkeypatch.setattr(
        "PIL.Image.open",
        lambda path, *a, **kw: Image.new("RGB", (5, 5), (0, 0, 255)),
    )

    result = module._get_image(row, "/tmp/extract", "ChartQA-X")
    assert isinstance(result, Image.Image)


def test_get_image_from_string_local_path(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    row = {"image": "./local/chart.png"}

    monkeypatch.setattr(os.path, "exists", lambda p: "local/chart.png" in p)
    monkeypatch.setattr(
        "PIL.Image.open",
        lambda path, *a, **kw: Image.new("RGB", (5, 5)),
    )

    result = module._get_image(row, "/tmp/extract", "ChartQA-X")
    assert isinstance(result, Image.Image)


def test_get_image_from_string_fallback_path(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    row = {"image": "./chart.png"}

    def fake_exists(path):
        if "ChartQA-X" in str(path):
            return True
        return False

    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(
        "PIL.Image.open",
        lambda path, *a, **kw: Image.new("RGB", (5, 5)),
    )

    result = module._get_image(row, "/tmp/extract", "ChartQA-X")
    assert isinstance(result, Image.Image)


def test_get_image_from_img_field(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    row = {"image": None, "img": "./chart.png"}

    def fake_exists(path):
        return "ChartQA-X" in str(path) and "chart.png" in str(path)

    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(
        "PIL.Image.open",
        lambda path, *a, **kw: Image.new("RGB", (5, 5)),
    )

    result = module._get_image(row, "/tmp/extract", "ChartQA-X")
    assert isinstance(result, Image.Image)


def test_get_image_raises_when_missing(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    row = {"image": None}

    monkeypatch.setattr(os.path, "exists", lambda p: False)

    with pytest.raises(ValueError, match="Row missing image"):
        module._get_image(row, "/tmp/extract", "ChartQA-X")


# ---------------------------------------------------------------------------
# _import_dataset
# ---------------------------------------------------------------------------

def test_import_dataset_inserts_rows(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, put_calls, _ = build_fake_aws()
    _install_fake_aws(module, fake_aws, monkeypatch)
    _install_fake_fs(monkeypatch)

    from .conftest import FakeCursor

    cursor = FakeCursor()
    pil_img = Image.new("RGB", (16, 16), (0, 255, 0))
    splits = {
        "train": [
            {
                "chart_type": "two_col",
                "QA": {"input": "What is shown?", "output": "A bar chart"},
                "image": pil_img,
            }
        ]
    }

    module._import_dataset(splits, cursor, "/tmp/extract", "ChartQA-X", max_rows=1)

    fake_aws.rds.insert_sample.assert_called_once()
    assert len(put_calls) == 1


def test_import_dataset_respects_max_rows(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    _install_fake_aws(module, fake_aws, monkeypatch)
    _install_fake_fs(monkeypatch)

    from .conftest import FakeCursor

    cursor = FakeCursor()
    rows = [
        {
            "chart_type": "multi_col",
            "QA": {"input": f"Q{i}?", "output": f"A{i}"},
            "image": Image.new("RGB", (10, 10)),
        }
        for i in range(5)
    ]

    module._import_dataset({"train": rows}, cursor, "/tmp/e", "ChartQA-X", max_rows=2)

    assert fake_aws.rds.insert_sample.call_count == 2


def test_import_dataset_skips_rows_with_missing_image(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    _install_fake_aws(module, fake_aws, monkeypatch)

    monkeypatch.setattr(os.path, "exists", lambda p: False)

    from .conftest import FakeCursor

    cursor = FakeCursor()
    row = {"chart_type": "x", "QA": {"input": "Q?", "output": "A"}, "image": None}

    module._import_dataset({"train": [row]}, cursor, "/tmp/e", "ChartQA-X", max_rows=None)

    fake_aws.rds.insert_sample.assert_not_called()


def test_import_dataset_extracts_qa_from_fallback_fields(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    _install_fake_aws(module, fake_aws, monkeypatch)
    _install_fake_fs(monkeypatch)

    from .conftest import FakeCursor

    cursor = FakeCursor()
    row = {
        "chart_type": "two_col",
        "QA": {},
        "question": "Fallback Q?",
        "answer": "Fallback A",
        "image": Image.new("RGB", (10, 10)),
    }

    module._import_dataset({"train": [row]}, cursor, "/tmp/e", "ChartQA-X", max_rows=1)

    args = fake_aws.rds.insert_sample.call_args[0]
    assert args[3] == "Fallback Q?"
    assert args[4] == "Fallback A"


# ---------------------------------------------------------------------------
# import_chartqax (top-level)
# ---------------------------------------------------------------------------

def test_import_chartqax_downloads_zip_when_env_set(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    _install_fake_aws(module, fake_aws, monkeypatch)

    pil_img = Image.new("RGB", (16, 16))
    splits = {"train": [{"chart_type": "x", "QA": {"input": "Q", "output": "A"}, "image": pil_img}]}

    hf_download_calls = []

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *a, **kw: splits
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda **kw: (hf_download_calls.append(kw), "/fake/zip")[1]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    monkeypatch.setenv("HF_IMAGE_ZIP", "images.zip")
    monkeypatch.setenv("HF_IMAGE_ROOT", "ChartQA-X")
    _install_fake_fs(monkeypatch)

    module.import_chartqax(max_rows=1)

    assert len(hf_download_calls) == 1
    assert hf_download_calls[0]["filename"] == "images.zip"


def test_import_chartqax_calls_create_table_and_processes(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, put_calls, _ = build_fake_aws()
    _install_fake_aws(module, fake_aws, monkeypatch)

    pil_img = Image.new("RGB", (16, 16), (0, 255, 0))
    splits = {
        "train": [
            {
                "chart_type": "two_col",
                "QA": {"input": "Q?", "output": "A"},
                "image": pil_img,
            }
        ]
    }
    _install_fake_hf(monkeypatch, splits)
    _install_fake_fs(monkeypatch)

    module.import_chartqax(max_rows=1)

    fake_aws.rds.create_table_if_not_exists.assert_called_once()
    fake_aws.rds.insert_sample.assert_called_once()
