import pytest
import numpy as np
from PIL import Image
from tests.helpers.torch_guard import require_torch

torch = require_torch()

from agdg.attack.methods.attackvlm import AttackVLMGraphTopic

class MockTextTargetModel:
    def __init__(self):
        self.device = "cpu"
    def get_image_size(self):
        return (512, 512)
    def embed_image(self, x, detach=False):
        feat = x.mean(dim=(-2, -1))
        feat = torch.nn.functional.pad(feat, (0, 509))
        return feat.detach() if detach else feat
    def embed_text(self, text):
        if isinstance(text, list):
            return torch.randn(len(text), 512)
        return torch.randn(1, 512)
    def __call__(self, a, b):
        # Handle batching
        if a.ndim == 2 and b.ndim == 2:
            if a.shape[0] == b.shape[0]:
                return (a * b).sum(dim=-1)
            elif b.shape[0] == 1:
                return (a * b).sum(dim=-1)
        return (a * b).sum()

def test_graph_topic_normalization():
    assert AttackVLMGraphTopic._normalize_topic("  Some   Topic  ") == "Some Topic"
    assert AttackVLMGraphTopic._normalize_topic(None) == ""

def test_graph_topic_anchor_terms():
    anchors = AttackVLMGraphTopic._topic_anchor_terms("A simple test about cats")
    assert "simple" in anchors
    assert "cats" in anchors
    assert "a" not in anchors
    assert len(anchors) <= 6

def test_graph_topic_build_anchor_phrases():
    phrases = AttackVLMGraphTopic._build_anchor_phrases("Cats")
    assert "Cats" in phrases
    assert "topic: Cats" in phrases
    assert "about cats" in phrases

def test_graph_topic_attack():
    model = MockTextTargetModel()
    attacker = AttackVLMGraphTopic(model, device="cpu")
    clean_img = Image.new("RGB", (100, 100), color="white")
    
    # Test build_topic_texts implicitly
    adv_img = attacker.attack(clean_img, "Cats", strength=0.5, hyperparameters={"steps": 1})
    assert isinstance(adv_img, Image.Image)
