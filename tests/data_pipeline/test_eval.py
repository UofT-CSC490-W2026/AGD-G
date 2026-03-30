import sys
from unittest.mock import MagicMock, patch

import pytest

mock_st = MagicMock()
sys.modules["sentence_transformers"] = mock_st

from agdg.data_pipeline.eval import evaluate_all


@pytest.fixture
def mock_aws():
    with patch("agdg.data_pipeline.eval.rds.get_db_connection") as mock_db, patch(
        "agdg.data_pipeline.eval.s3.get_image", return_value=b"png"
    ):
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        mock_db.return_value.__enter__.return_value = conn
        yield {"cur": cur, "conn": conn}


@pytest.fixture
def mock_models():
    sim_model = MagicMock()
    mock_st.SentenceTransformer.return_value = sim_model

    with patch("agdg.data_pipeline.eval.get_device", return_value="cpu"), patch(
        "agdg.data_pipeline.eval.load_vlm", return_value=(MagicMock(), MagicMock(), "cpu", "float32")
    ), patch(
        "agdg.data_pipeline.eval.generate_image_response", return_value="Target text"
    ):
        yield {"sim_model": sim_model}


def test_evaluate_all_no_rows(mock_aws, mock_models):
    mock_aws["cur"].fetchall.return_value = []
    result = evaluate_all()
    assert result == {"evaluated": 0}


def test_evaluate_all_success(mock_aws, mock_models):
    mock_aws["cur"].fetchall.return_value = [(1, "Clean caption", "Target text", "chart-uuid")]

    with patch("agdg.data_pipeline.eval._similarity_scores", return_value=(0.1, 0.9)):
        result = evaluate_all()

    assert result["evaluated"] == 1
    assert result["succeeded"] == 1
    assert result["winners"]["B"] == 1

    update_calls = [call for call in mock_aws["cur"].execute.call_args_list if "INSERT INTO adversarial_answers" in call[0][0]]
    assert len(update_calls) == 1
    assert update_calls[0][0][1][0:4] == (1, "llava-hf/llava-1.5-7b-hf", "Target text", True)


def test_evaluate_all_missing_data(mock_aws, mock_models):
    with patch("agdg.data_pipeline.eval.generate_image_response", side_effect=Exception("bad image")):
        mock_aws["cur"].fetchall.return_value = [(1, "Clean", "Target", "chart-uuid")]
        result = evaluate_all()
    assert result["evaluated"] == 0


def test_evaluate_all_tie(mock_aws, mock_models):
    mock_aws["cur"].fetchall.return_value = [(1, "A", "B", "chart-uuid")]

    with patch("agdg.data_pipeline.eval._similarity_scores", return_value=(0.8, 0.8)):
        result = evaluate_all()

    assert result["winners"]["Tie"] == 1


def test_similarity_scores_direct():
    torch = pytest.importorskip("torch")
    from agdg.data_pipeline.eval import _similarity_scores

    model = MagicMock()
    model.encode.return_value = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    score_a, score_b = _similarity_scores(model, "output", "text_a", "text_b")

    assert score_a == pytest.approx(1.0)
    assert score_b == pytest.approx(0.0)


def test_evaluate_all_with_strategy(mock_aws, mock_models):
    mock_aws["cur"].fetchall.return_value = []
    result = evaluate_all(target_strategy="qwen")
    assert result == {"evaluated": 0}
    sql_call = mock_aws["cur"].execute.call_args
    assert "AND ta.target_strategy" in sql_call[0][0]
    assert "qwen" in sql_call[0][1]


def test_evaluate_all_with_max_rows(mock_aws, mock_models):
    mock_aws["cur"].fetchall.return_value = []
    result = evaluate_all(max_rows=5)
    assert result == {"evaluated": 0}
    sql_call = mock_aws["cur"].execute.call_args
    assert "LIMIT" in sql_call[0][0]
    assert 5 in sql_call[0][1]


def test_evaluate_all_batch_commit(mock_aws, mock_models):
    mock_aws["cur"].fetchall.return_value = [(1, "Clean", "Target", "chart-uuid")]

    with patch("agdg.data_pipeline.eval._similarity_scores", return_value=(0.1, 0.9)), \
         patch("agdg.data_pipeline.eval.BATCH_SIZE", 1):
        result = evaluate_all()

    assert result["evaluated"] == 1
    assert mock_aws["conn"].commit.called
