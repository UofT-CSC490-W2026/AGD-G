from agdg.data_pipeline.chart_type import ChartType


def test_graph_type_string_and_names():
    assert str(ChartType.BAR) == "BAR"
    names = ChartType.get_names()
    assert "'BAR'" in names
    assert "'TREEMAP'" in names
    assert names.count("'") == len(ChartType) * 2
