
import pytest
import torch
import sys
from unittest.mock import MagicMock, patch, ANY

# Mock sentence_transformers at the sys.modules level to avoid Keras errors during import
mock_st = MagicMock()
sys.modules["sentence_transformers"] = mock_st

from agdg.data_pipeline.eval import evaluate_all

@pytest.fixture
def mock_aws():
    with patch("agdg.data_pipeline.eval.get_db_connection") as mock_db:
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        mock_db.return_value.__enter__.return_value = conn
        yield {"db": mock_db, "cur": cur, "conn": conn}

@pytest.fixture
def mock_sim_model():
    mock_model = MagicMock()
    # Mock embeddings: 3 vectors (output, A, B)
    mock_model.encode.return_value = torch.tensor([
        [1.0, 0.0], # output
        [0.0, 1.0], # A (clean)
        [1.0, 0.0]  # B (target)
    ])
    mock_st.SentenceTransformer.return_value = mock_model
    yield mock_model

def test_evaluate_all_no_rows(mock_aws, mock_sim_model):
    mock_aws["cur"].fetchall.return_value = []
    result = evaluate_all()
    assert result == {"evaluated": 0}

def test_evaluate_all_success(mock_aws, mock_sim_model):
    # Mock data: output matches target (B) better than clean (A)
    mock_aws["cur"].fetchall.return_value = [
        (1, "Clean caption", "Target text", "Target text")
    ]
    
    result = evaluate_all()
    
    assert result["evaluated"] == 1
    assert result["winners"]["B"] == 1
    
    # Verify DB update
    update_calls = [call for call in mock_aws["cur"].execute.call_args_list if "UPDATE samples" in call[0][0]]
    assert len(update_calls) == 1
    assert update_calls[0][0][1] == ("B", 1)

def test_evaluate_all_missing_data(mock_aws, mock_sim_model):
    mock_aws["cur"].fetchall.return_value = [
        (1, None, "Target", "Response")
    ]
    
    result = evaluate_all()
    assert result["evaluated"] == 0

def test_evaluate_all_tie(mock_aws, mock_sim_model):
    # Mock embeddings to be identical
    mock_sim_model.encode.return_value = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    
    mock_aws["cur"].fetchall.return_value = [
        (1, "A", "B", "Output")
    ]
    
    result = evaluate_all()
    assert result["winners"]["Tie"] == 1
