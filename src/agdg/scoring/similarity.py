"""Sentence-similarity helpers for comparing VLM outputs against reference texts."""


def determine_winner(
    score_a: float,
    score_b: float,
    cutoff_score: float = 0.5,
    margin: float = 0.02,
    drift_weight: float = 0.7,
    target_weight: float = 0.3,
) -> str:
    """
    Determines the winning text based on similarity scores.

    The primary signal is whether the adversarial output has drifted away from
    the clean answer (low score_a). Closeness to the target caption (high score_b)
    adds a bonus but is not required for a successful attack.

    attack_score = drift_weight * (1 - score_a) + target_weight * score_b
    clean_score  = drift_weight * score_a        + target_weight * (1 - score_b)

    Returns "Neither" if attack_score < cutoff_score (drift was not meaningful).
    Returns "Tie" if the two composite scores are within margin of each other.
    Returns "B" if attack_score > clean_score (attack succeeded).
    Returns "A" otherwise (output stayed close to clean).

    >>> determine_winner(0.90, 0.50)
    'A'
    >>> determine_winner(0.50, 0.90)
    'B'
    >>> determine_winner(0.30, 0.40)
    'B'
    >>> determine_winner(0.60, 0.70, cutoff_score=0.75)
    'Neither'
    >>> determine_winner(0.85, 0.86, margin=0.02)
    'Tie'
    >>> determine_winner(0.85, 0.82, margin=0.02)
    'A'
    """
    attack_score = drift_weight * (1 - score_a) + target_weight * score_b
    clean_score  = drift_weight * score_a        + target_weight * (1 - score_b)

    if attack_score < cutoff_score:
        return "Neither"

    if abs(attack_score - clean_score) <= margin:
        return "Tie"

    return "B" if attack_score > clean_score else "A"

def get_device():
    """
    Returns the CUDA device if available, then MPS (Apple Silicon), otherwise falls back to CPU.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def evaluate_similarity(model, output_text: str, text_a: str, text_b: str, cutoff_score: float = 0.5, margin: float = 0.02) -> str:
    """
    Generates embeddings and evaluates text similarity.
    Expects an already-initialized SentenceTransformer model to be passed in.
    """
    import torch.nn.functional as F

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
    import doctest # pragma: no cover
    doctest.testmod(verbose=True) # pragma: no cover
