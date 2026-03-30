import pytest
import numpy as np
from PIL import Image
from collections.abc import Sequence
from tests.helpers.torch_guard import require_torch

torch = require_torch()

from agdg.attack.methods.attackvlm import (
    AttackVLMGraphTopic,
    AttackVLMImage,
    AttackVLMOCR,
    AttackVLMText,
    AttackVLMUntargeted,
)


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
        if a.ndim == 2 and b.ndim == 2:
            if a.shape[0] == b.shape[0]:
                return (a * b).sum(dim=-1)
            elif b.shape[0] == 1:
                return (a * b).sum(dim=-1)
        return (a * b).sum()


class MockImageTargetModel:
    def __init__(self):
        self.device = "cpu"
    def get_image_size(self):
        return (512, 512)
    def embed_image(self, x, detach=False):
        feat = x.mean(dim=(-2, -1))
        feat = torch.nn.functional.pad(feat, (0, 509))
        return feat.detach() if detach else feat
    def __call__(self, a, b):
        return (a * b).sum()


# ---------------------------------------------------------------------------
# Existing GraphTopic tests
# ---------------------------------------------------------------------------

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
    adv_img = attacker.attack(clean_img, "Cats", strength=0.5, hyperparameters={"steps": 1})
    assert isinstance(adv_img, Image.Image)


# ---------------------------------------------------------------------------
# _run_attack None guards (lines 115, 120)
# ---------------------------------------------------------------------------

def test_run_attack_none_clean():
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    with pytest.raises(TypeError, match="must not be None"):
        attacker._run_attack(clean=None, eps=0.1, alpha=0.01, steps=0)


def test_run_attack_none_in_list():
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    with pytest.raises(TypeError, match="must not be None"):
        attacker._run_attack(
            clean=[Image.new("RGB", (10, 10)), None],
            eps=0.1, alpha=0.01, steps=0,
        )


# ---------------------------------------------------------------------------
# Empty-list guard on each .attack() variant (lines 288, 374, 574, 627)
# ---------------------------------------------------------------------------

def test_image_attack_empty_list():
    model = MockImageTargetModel()
    attacker = AttackVLMImage(model, device="cpu")
    assert attacker.attack(clean=[], target=Image.new("RGB", (10, 10)), strength=0.5) == []


def test_text_attack_empty_list():
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")
    assert attacker.attack(clean=[], target="cats", strength=0.5) == []


def test_graph_topic_attack_empty_list():
    model = MockTextTargetModel()
    attacker = AttackVLMGraphTopic(model, device="cpu")
    assert attacker.attack(clean=[], target="cats", strength=0.5) == []


def test_ocr_attack_empty_list():
    model = MockTextTargetModel()
    attacker = AttackVLMOCR(model, device="cpu")
    assert attacker.attack(clean=[], target="cats", strength=0.5) == []


# ---------------------------------------------------------------------------
# _expand_loss branches (lines 335-337)
# ---------------------------------------------------------------------------

def test_expand_loss_single_element():
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")
    result = attacker._expand_loss(torch.tensor([5.0]), batch_size=3)
    assert result.shape == (3,)
    assert (result == 5.0).all()


def test_expand_loss_reshape_fallback():
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")
    result = attacker._expand_loss(torch.tensor([1.0, 2.0, 3.0, 4.0]), batch_size=2)
    assert result.shape == (2,)
    assert result[0] == 1.0 and result[1] == 2.0


# ---------------------------------------------------------------------------
# _build_anchor_phrases dedup branch (line 444)
# ---------------------------------------------------------------------------

def test_build_anchor_phrases_dedup():
    phrases = AttackVLMGraphTopic._build_anchor_phrases("cats")
    assert len(phrases) == len(set(phrases))
    assert "cats" in phrases


# ---------------------------------------------------------------------------
# GraphTopic batch target_text setup (lines 508-513)
# ---------------------------------------------------------------------------

def test_graph_topic_batch_target():
    model = MockTextTargetModel()
    attacker = AttackVLMGraphTopic(model, device="cpu")
    imgs = [Image.new("RGB", (100, 100), color="white")] * 2
    adv = attacker.attack(imgs, ["cats", "dogs"], strength=0.5, hyperparameters={"steps": 1})
    assert isinstance(adv, Sequence)
    assert len(adv) == 2


# ---------------------------------------------------------------------------
# GraphTopic with source_topic (lines 529-544, 559-560, 562-563)
# ---------------------------------------------------------------------------

def test_graph_topic_with_source_topic():
    model = MockTextTargetModel()
    attacker = AttackVLMGraphTopic(model, device="cpu")
    clean_img = Image.new("RGB", (100, 100), color="white")
    adv = attacker.attack(
        clean_img,
        "cats",
        strength=0.5,
        hyperparameters={"steps": 1, "source_topic": "dogs"},
    )
    assert isinstance(adv, Image.Image)


def test_graph_topic_with_batch_source_topic():
    model = MockTextTargetModel()
    attacker = AttackVLMGraphTopic(model, device="cpu")
    imgs = [Image.new("RGB", (100, 100), color="white")] * 2
    adv = attacker.attack(
        imgs,
        ["cats", "dogs"],
        strength=0.5,
        hyperparameters={"steps": 1, "source_topic": ["fish", "birds"]},
    )
    assert isinstance(adv, Sequence)
    assert len(adv) == 2


# ---------------------------------------------------------------------------
# OCR setup with 3D tensor (line 609)
# ---------------------------------------------------------------------------

def test_ocr_setup_3d_tensor():
    model = MockTextTargetModel()
    attacker = AttackVLMOCR(model, device="cpu")
    clean_t = torch.rand(3, 64, 64)
    attacker.setup(clean_t, target_text="cats")
    assert attacker.perturbation_mask is not None
    assert attacker.perturbation_mask.ndim == 4
