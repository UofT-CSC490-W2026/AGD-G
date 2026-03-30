"""Tests for agdg.data_pipeline.import_chartx."""
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
    return importlib.import_module("agdg.data_pipeline.import_chartx")


# ---------------------------------------------------------------------------
# HuggingFace / filesystem fakes
# ---------------------------------------------------------------------------

_real_image_open = Image.open


def _install_fake_hf(monkeypatch, splits):
    """
    *splits* is a dict of split_name → list[row].  Each row needs
    ``chart_type``, ``img``, and ``QA`` with ``input``/``output``.
    """
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *a, **kw: splits
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda **kw: "/fake/ChartX_png.zip"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)


def _install_fake_fs(monkeypatch, *, fail_open=False):
    """Patch os/zipfile and PIL.Image.open to avoid real disk I/O."""

    class _FakeZipFile:
        def __init__(self, *a, **kw):
            pass
        def extractall(self, path):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    def fake_image_open(path, *args, **kwargs):
        if isinstance(path, str) and "ChartX_png" in path:
            if fail_open:
                raise FileNotFoundError(path)
            return Image.new("RGB", (32, 32), (255, 0, 0))
        return _real_image_open(path, *args, **kwargs)

    monkeypatch.setattr("zipfile.ZipFile", _FakeZipFile)
    monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
    monkeypatch.setattr("PIL.Image.open", fake_image_open)


def _make_row(chart_type="bar_chart", img="./test/chart1.png", q="Q?", a="A"):
    return {
        "chart_type": chart_type,
        "img": img,
        "QA": {"input": q, "output": a},
    }


# ---------------------------------------------------------------------------
# chart_type_to_graph_type (pure function)
# ---------------------------------------------------------------------------

def test_chart_type_to_graph_type_known_types(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    from agdg.data_pipeline.schema import GraphType

    assert module.chart_type_to_graph_type("bar_chart") == GraphType.BAR
    assert module.chart_type_to_graph_type("line_chart") == GraphType.LINE
    assert module.chart_type_to_graph_type("pie_chart") == GraphType.PIE
    assert module.chart_type_to_graph_type("heatmap") == GraphType.HEATMAP
    assert module.chart_type_to_graph_type("radar") == GraphType.RADAR
    assert module.chart_type_to_graph_type("treemap") == GraphType.TREEMAP
    assert module.chart_type_to_graph_type("3D-Bar") == GraphType.THREE_D
    assert module.chart_type_to_graph_type("candlestick") == GraphType.CANDLE
    assert module.chart_type_to_graph_type("bubble") == GraphType.SCATTER


def test_chart_type_to_graph_type_unknown_falls_back(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    from agdg.data_pipeline.schema import GraphType

    assert module.chart_type_to_graph_type("unknown_xyz") == GraphType.OTHER


# ---------------------------------------------------------------------------
# import_chartx tests
# ---------------------------------------------------------------------------

def test_import_inserts_rows_and_commits(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, put_calls, conn = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)
    _install_fake_hf(monkeypatch, {"train": [_make_row()]})
    _install_fake_fs(monkeypatch)

    module.import_chartx(max_rows=1)

    fake_aws.create_table_if_not_exists.assert_called_once()
    fake_aws.add_sample.assert_called_once()
    args = fake_aws.add_sample.call_args[0]
    assert args[1] == "ChartX"
    assert args[3] == "Q?"
    assert args[4] == "A"
    assert len(put_calls) == 1
    assert conn.commits >= 1


def test_import_respects_max_rows(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)
    rows = [_make_row(img=f"./test/chart{i}.png") for i in range(5)]
    _install_fake_hf(monkeypatch, {"train": rows})
    _install_fake_fs(monkeypatch)

    module.import_chartx(max_rows=2)

    assert fake_aws.add_sample.call_count == 2


def test_import_none_max_rows_processes_all(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)
    rows = [_make_row(img=f"./test/chart{i}.png") for i in range(3)]
    _install_fake_hf(monkeypatch, {"train": rows})
    _install_fake_fs(monkeypatch)

    module.import_chartx(max_rows=None)

    assert fake_aws.add_sample.call_count == 3


def test_import_counts_failures_and_continues(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)

    good_row = _make_row(img="./test/good.png")
    bad_row = {"chart_type": "bar_chart", "img": None, "QA": None}
    _install_fake_hf(monkeypatch, {"train": [bad_row, good_row]})
    _install_fake_fs(monkeypatch)

    module.import_chartx(max_rows=2)

    assert fake_aws.add_sample.call_count == 1


def test_import_stops_after_max_failures(monkeypatch, fake_boto):
    import pytest

    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)

    bad_row = {"chart_type": "x", "img": None, "QA": None}
    _install_fake_hf(monkeypatch, {"train": [bad_row] * 51})
    _install_fake_fs(monkeypatch)

    with pytest.raises(RuntimeError, match="50 failures"):
        module.import_chartx(max_rows=None)


def test_import_processes_multiple_splits(monkeypatch, fake_boto):
    module = _load(monkeypatch, fake_boto)
    fake_aws, _, _ = build_fake_aws()
    monkeypatch.setattr(module, "aws", fake_aws)
    _install_fake_hf(monkeypatch, {
        "train": [_make_row(img="./test/t1.png")],
        "test": [_make_row(img="./test/t2.png")],
    })
    _install_fake_fs(monkeypatch)

    module.import_chartx(max_rows=None)

    assert fake_aws.add_sample.call_count == 2
