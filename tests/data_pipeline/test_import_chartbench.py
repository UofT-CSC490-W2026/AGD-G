"""Tests for agdg.data_pipeline.import_chartbench."""
import builtins
import importlib
import io
import os
import sys
import types

from PIL import Image

from .conftest import build_fake_aws, make_png


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def _load(monkeypatch, fake_boto):
    for mod in list(sys.modules):
        if mod.startswith("agdg.data_pipeline.") and mod != "agdg.data_pipeline":
            sys.modules.pop(mod, None)
    return importlib.import_module("agdg.data_pipeline.import_chartbench")


# ---------------------------------------------------------------------------
# Filesystem / HuggingFace fakes
# ---------------------------------------------------------------------------

_real_exists = os.path.exists
_real_open = builtins.open


def _install_fake_hf(monkeypatch, dataset_rows):
    """Inject fake ``datasets`` and ``huggingface_hub`` modules."""
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *a, **kw: dataset_rows
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda **kw: "/fake/test.zip"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)


def _install_fake_fs(monkeypatch, *, missing_paths=None):
    """Patch os/zipfile/shutil so no real disk I/O happens."""
    missing = set(missing_paths or [])

    def fake_exists(path):
        path_str = str(path)
        if "chartbench_extracted" in path_str:
            return path_str not in missing
        return _real_exists(path_str)

    class _FakeFile:
        def __init__(self, data):
            self._data = data
        def read(self):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    def fake_open(path, mode="r", **kwargs):
        if "chartbench_extracted" in str(path) and "b" in mode:
            return _FakeFile(make_png())
        return _real_open(path, mode, **kwargs)

    class _FakeZipFile:
        def __init__(self, *a, **kw):
            pass
        def extractall(self, path):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr("zipfile.ZipFile", _FakeZipFile)
    monkeypatch.setattr("shutil.rmtree", lambda *a, **kw: None)


def _make_row(image="./data/test/img.png", chart_type="bar", query="Q?", label="A"):
    return {
        "image": image,
        "type": {"chart": chart_type},
        "conversation": [{"query": query, "label": label}],
    }


# ---------------------------------------------------------------------------
# Type-map tests
# ---------------------------------------------------------------------------

def test_type_map_covers_expected_chart_types(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    from agdg.data_pipeline.schema import GraphType

    expected = {
        "area": GraphType.AREA,
        "bar": GraphType.BAR,
        "box": GraphType.BOX,
        "combination": GraphType.OTHER,
        "line": GraphType.LINE,
        "node_link": GraphType.NODE,
        "pie": GraphType.PIE,
        "radar": GraphType.RADAR,
        "scatter": GraphType.SCATTER,
    }
    assert module.CHARTBENCH_TYPE_MAP == expected


# ---------------------------------------------------------------------------
# import_chartbench tests
# ---------------------------------------------------------------------------

def test_import_inserts_rows_and_returns_summary(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, put_calls, _ = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)
    _install_fake_hf(monkeypatch, [_make_row()])
    _install_fake_fs(monkeypatch)

    result = module.import_chartbench(max_rows=1)

    assert result == {
        "total_rows": 1,
        "rows_inserted": 1,
        "rows_skipped": 0,
        "unique_images": 1,
    }
    fake_aws.create_table_if_not_exists.assert_called_once()
    fake_aws.add_sample.assert_called_once()
    args = fake_aws.add_sample.call_args[0]
    assert args[1] == "ChartBench"
    assert args[2] == "BAR"
    assert args[3] == "Q?"
    assert args[4] == "A"


def test_import_with_clean_wipes_first(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)
    _install_fake_hf(monkeypatch, [])
    _install_fake_fs(monkeypatch)

    module.import_chartbench(max_rows=0, clean=True)

    fake_aws.wipe_s3.assert_called_once()
    fake_aws.wipe_rds.assert_called_once()


def test_import_without_clean_skips_wipe(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)
    _install_fake_hf(monkeypatch, [])
    _install_fake_fs(monkeypatch)

    module.import_chartbench(max_rows=0, clean=False)

    fake_aws.wipe_s3.assert_not_called()
    fake_aws.wipe_rds.assert_not_called()


def test_import_respects_max_rows(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)
    rows = [_make_row(image=f"./data/test/img{i}.png") for i in range(5)]
    _install_fake_hf(monkeypatch, rows)
    _install_fake_fs(monkeypatch)

    result = module.import_chartbench(max_rows=2)

    assert result["rows_inserted"] == 2
    assert result["total_rows"] == 2


def test_import_deduplicates_images(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, put_calls, _ = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)
    rows = [
        _make_row(image="./data/test/same.png", query="Q1"),
        _make_row(image="./data/test/same.png", query="Q2"),
    ]
    _install_fake_hf(monkeypatch, rows)
    _install_fake_fs(monkeypatch)

    result = module.import_chartbench(max_rows=2)

    assert result["rows_inserted"] == 2
    assert result["unique_images"] == 1
    assert len(put_calls) == 1


def test_import_maps_unknown_type_to_other(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)
    _install_fake_hf(monkeypatch, [_make_row(chart_type="alien_chart")])
    _install_fake_fs(monkeypatch)

    result = module.import_chartbench(max_rows=1)

    assert result["rows_inserted"] == 1
    assert fake_aws.add_sample.call_args[0][2] == "OTHER"


def test_import_skips_missing_images(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)
    _install_fake_hf(monkeypatch, [_make_row(image="./data/test/gone.png")])
    _install_fake_fs(
        monkeypatch,
        missing_paths=["/tmp/chartbench_extracted/test/gone.png"],
    )

    result = module.import_chartbench(max_rows=1)

    assert result["rows_skipped"] == 1
    assert result["rows_inserted"] == 0
    fake_aws.add_sample.assert_not_called()


def test_import_commits_in_batches(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, conn = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)
    monkeypatch.setattr(module, "BATCH_SIZE", 2)
    rows = [_make_row(image=f"./data/test/img{i}.png") for i in range(4)]
    _install_fake_hf(monkeypatch, rows)
    _install_fake_fs(monkeypatch)

    result = module.import_chartbench(max_rows=4)

    assert result["rows_inserted"] == 4
    assert conn.commits >= 2
