import torch
import torch.nn.functional as F

def determine_winner(score_a: float, score_b: float, cutoff_score: float = 0.75, margin: float = 0.02) -> str:
    """
    Determines the winning text based on similarity scores.

    >>> determine_winner(0.90, 0.50)
    'A'
    >>> determine_winner(0.50, 0.90)
    'B'
    >>> determine_winner(0.60, 0.70, cutoff_score=0.75)
    'Neither'
    >>> determine_winner(0.85, 0.86, margin=0.02)
    'Tie'
    >>> determine_winner(0.85, 0.82, margin=0.02)
    'A'
    """
    if score_a < cutoff_score and score_b < cutoff_score:
        return "Neither"
        
    if abs(score_a - score_b) <= margin:
        return "Tie"
        
    if score_a > score_b:
        return "A"
    else:
        return "B"

def get_device() -> torch.device:
    """
    Returns the CUDA device if available, then MPS (Apple Silicon), otherwise falls back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def evaluate_similarity(model, output_text: str, text_a: str, text_b: str, cutoff_score: float = 0.75, margin: float = 0.02) -> str:
    """
    Generates embeddings and evaluates text similarity.
    Expects an already-initialized SentenceTransformer model to be passed in.
    """
    embeddings = model.encode([output_text, text_a, text_b], convert_to_tensor=True)
    
    # Separate the tensors and add a batch dimension for the PyTorch function
    output_vec = embeddings[0].unsqueeze(0)
    vec_a = embeddings[1].unsqueeze(0)
    vec_b = embeddings[2].unsqueeze(0)
    
    # Calculate cosine similarity natively on the given device
    score_a = F.cosine_similarity(output_vec, vec_a).item()
    score_b = F.cosine_similarity(output_vec, vec_b).item()
    
    return determine_winner(score_a, score_b, cutoff_score, margin)

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)