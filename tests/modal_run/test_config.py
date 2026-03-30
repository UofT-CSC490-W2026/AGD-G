from .helpers import ensure_modal_root, import_fresh


def test_graph_type_string_and_names():
    ensure_modal_root()
    config = import_fresh("agdg.data_pipeline.config")

    assert str(config.GraphType.BAR) == "BAR"
    names = config.GraphType.get_names()
    assert "'BAR'" in names
    assert "'TREEMAP'" in names
    assert names.count("'") == len(config.GraphType) * 2
