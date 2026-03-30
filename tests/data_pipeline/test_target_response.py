"""Tests for agdg.data_pipeline.target_response."""
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from .conftest import make_png


@pytest.fixture
def mock_aws():
    with patch("agdg.data_pipeline.target_response.rds") as mock_rds, \
         patch("agdg.data_pipeline.target_response.s3") as mock_s3:
        conn = MagicMock()
        cur = MagicMock()
        cur.rowcount = 1
        conn.cursor.return_value.__enter__.return_value = cur
        mock_rds.get_db_connection.return_value.__enter__.return_value = conn
        mock_s3.get_image.return_value = make_png()
        yield {"rds": mock_rds, "s3": mock_s3, "conn": conn, "cur": cur}


@pytest.fixture
def mock_targeter():
    targeter = MagicMock()
    targeter.return_value = ["target caption"]
    targeter.generate_raw.return_value = [{"target": "caption", "thinking": "reasoning"}]
    with patch(
        "agdg.data_pipeline.target_response.build_targeting_strategy",
        return_value=targeter,
    ):
        yield targeter


def test_generate_no_rows(mock_aws, mock_targeter):
    from agdg.data_pipeline.target_response import generate_target_responses

    mock_aws["rds"].iter_target_inputs.return_value = iter([])

    result = generate_target_responses("qwen")

    assert result == {"processed": 0}


def test_generate_processes_rows(mock_aws, mock_targeter):
    from agdg.data_pipeline.target_response import generate_target_responses

    mock_aws["rds"].iter_target_inputs.return_value = iter([
        {"clean_answer_id": 1, "clean_answer": "clean", "clean_chart": "uuid1"},
    ])

    result = generate_target_responses("qwen")

    assert result["processed"] == 1
    mock_aws["rds"].insert_target_answer.assert_called_once()


def test_generate_respects_max_rows(mock_aws, mock_targeter):
    from agdg.data_pipeline.target_response import generate_target_responses

    mock_aws["rds"].iter_target_inputs.return_value = iter([
        {"clean_answer_id": 1, "clean_answer": "c1", "clean_chart": "u1"},
        {"clean_answer_id": 2, "clean_answer": "c2", "clean_chart": "u2"},
    ])

    result = generate_target_responses("qwen", max_rows=1)

    assert result["processed"] == 1


def test_generate_batch_commit(mock_aws, mock_targeter):
    from agdg.data_pipeline.target_response import generate_target_responses

    mock_aws["rds"].iter_target_inputs.return_value = iter([
        {"clean_answer_id": 1, "clean_answer": "c1", "clean_chart": "u1"},
    ])

    result = generate_target_responses("qwen", batch_size=1)

    assert result["processed"] == 1
    mock_aws["conn"].commit.assert_called()


def test_generate_handles_error(mock_aws, mock_targeter):
    from agdg.data_pipeline.target_response import generate_target_responses

    mock_targeter.return_value = None
    mock_targeter.side_effect = RuntimeError("model failed")

    mock_aws["rds"].iter_target_inputs.return_value = iter([
        {"clean_answer_id": 1, "clean_answer": "c1", "clean_chart": "u1"},
    ])

    result = generate_target_responses("qwen")

    assert result["processed"] == 0


def test_generate_passes_source(mock_aws, mock_targeter):
    from agdg.data_pipeline.target_response import generate_target_responses

    mock_aws["rds"].iter_target_inputs.return_value = iter([])

    generate_target_responses("qwen", source="ChartBench")

    mock_aws["rds"].iter_target_inputs.assert_called_once_with(
        mock_aws["conn"], "qwen", source="ChartBench"
    )


def test_preview_returns_results(mock_aws, mock_targeter):
    from agdg.data_pipeline.target_response import preview_target_responses

    mock_aws["rds"].iter_target_inputs_sampled.return_value = iter([
        {"clean_answer": "clean", "clean_chart": "u1", "chart_source": "CB"},
    ])

    results = preview_target_responses("qwen", per_source=5)

    assert len(results) == 1
    assert results[0]["source"] == "CB"
    assert results[0]["target"] == "caption"
    assert results[0]["thinking"] == "reasoning"


def test_preview_handles_error(mock_aws, mock_targeter):
    from agdg.data_pipeline.target_response import preview_target_responses

    mock_targeter.generate_raw.side_effect = RuntimeError("fail")
    mock_aws["rds"].iter_target_inputs_sampled.return_value = iter([
        {"clean_answer": "clean", "clean_chart": "u1", "chart_source": "CB"},
    ])

    results = preview_target_responses("qwen")

    assert results == []


def test_preview_no_rows(mock_aws, mock_targeter):
    from agdg.data_pipeline.target_response import preview_target_responses

    mock_aws["rds"].iter_target_inputs_sampled.return_value = iter([])

    results = preview_target_responses("qwen")

    assert results == []
