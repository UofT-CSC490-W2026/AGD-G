from agdg.data_pipeline.schema import GraphType


def test_graph_type_string_and_names():
    assert str(GraphType.BAR) == "BAR"
    names = GraphType.get_names()
    assert "'BAR'" in names
    assert "'TREEMAP'" in names
    assert names.count("'") == len(GraphType) * 2
