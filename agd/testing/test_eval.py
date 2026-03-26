import pytest
from unittest.mock import MagicMock
import torch
import os
from sentence_transformers import SentenceTransformer
from agd.scoring.eval import determine_winner, evaluate_similarity, get_device

# --- Tests for the Core Mathematical Logic ---

def test_determine_winner_clear_a():
    assert determine_winner(0.85, 0.50, cutoff_score=0.70) == "A"

def test_determine_winner_clear_b():
    assert determine_winner(0.50, 0.85, cutoff_score=0.70) == "B"

def test_determine_winner_neither():
    # Both are below the 0.75 threshold
    assert determine_winner(0.74, 0.73, cutoff_score=0.75) == "Neither"

def test_determine_winner_tie_exact():
    assert determine_winner(0.90, 0.90, margin=0.02) == "Tie"

def test_determine_winner_tie_within_margin():
    # Difference is 0.01, which is less than the 0.02 margin
    assert determine_winner(0.91, 0.90, margin=0.02) == "Tie"

def test_determine_winner_just_outside_margin():
    # Difference is 0.03, which is greater than the 0.02 margin
    assert determine_winner(0.93, 0.90, margin=0.02) == "A"


# --- Tests for the Pipeline (Mocking the Model) ---

@pytest.fixture
def mock_embedding_model():
    """
    Creates a fake model that returns predictable tensors instead of 
    running actual GPU calculations.
    """
    mock_model = MagicMock()
    
    # Create fake 1D tensors to represent the embeddings
    tensor_output = torch.tensor([1.0, 0.0, 0.0])
    tensor_a = torch.tensor([1.0, 0.0, 0.0]) # Perfect match with output
    tensor_b = torch.tensor([0.0, 1.0, 0.0]) # No match
    
    # When model.encode is called, return this stack of tensors
    mock_model.encode.return_value = torch.stack([tensor_output, tensor_a, tensor_b])
    return mock_model

def test_evaluate_similarity_pipeline(mock_embedding_model):
    """
    Tests that the vectors are passed correctly and the expected 
    winner is returned without invoking the real Jina model.
    """
    result = evaluate_similarity(
        model=mock_embedding_model,
        output_text="Test output",
        text_a="Perfect match",
        text_b="Terrible match",
        cutoff_score=0.50
    )
    
    # Because tensor_output and tensor_a are identical, 'A' should win
    assert result == "A"
    
    # Verify the model was called with the correct arguments
    mock_embedding_model.encode.assert_called_once()
    args, kwargs = mock_embedding_model.encode.call_args
    assert args[0] == ["Test output", "Perfect match", "Terrible match"]
    assert kwargs["convert_to_tensor"] is True

def test_evaluate_similarity_real_model():
    """
    Tests the pipeline using the actual Jina model.
    This will download the model if it is not already cached.
    Works across CUDA, Apple Silicon (MPS), and CPU.
    """
    # Fix for Apple Silicon (MPS) missing operators in PyTorch
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Determine the best available hardware device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v5-text-small-text-matching", 
        device=device,
        trust_remote_code=True
    )
    
    result = evaluate_similarity(
        model=model,
        output_text="The network went down.",
        text_a="There was a server outage.",
        text_b="The food was cold.",
        cutoff_score=0.50
    )
    
    assert result == "A"