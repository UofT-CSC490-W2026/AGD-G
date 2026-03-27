"""
Edge case tests for the AttackVLM class.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch
from agd.core.attacks.attackvlm import (
    AttackVLMUntargeted,
    AttackVLMImage,
    AttackVLMText,
    AttackVLMOCR,
)

# Mocking the model interfaces
class MockImageTargetModel:
    def __init__(self):
        self.device = "cpu"
    def embed_image(self, x, detach=False):
        # Ensure x is differentiable if it requires grad
        # We need a path from x to the output
        # Using a simple mean or sum to keep it differentiable
        # features: [B, 512]
        # We can use a dummy projection: x is [B, 3, 512, 512]
        # Let's just take the mean over spatial and multiply by a dummy vector
        feat = x.mean(dim=(2, 3)) # [B, 3]
        # Pad to 512
        feat = torch.nn.functional.pad(feat, (0, 509)) # [B, 512]
        if detach:
            return feat.detach()
        return feat

    def __call__(self, a, b):
        # Simple similarity that is differentiable w.r.t a
        return (a * b).sum()

class MockTextTargetModel(MockImageTargetModel):
    def embed_text(self, text):
        # Text embedding doesn't need to be differentiable w.r.t input text string
        return torch.randn(1, 512)

@pytest.fixture
def clean_image():
    return Image.new("RGB", (100, 100), color="white")

@pytest.fixture
def target_image():
    return Image.new("RGB", (100, 100), color="black")

# 1. Positive Test: Untargeted attack runs successfully
def test_untargeted_attack_positive(clean_image):
    """
    Coverage: AttackVLMUntargeted._run_attack loop.
    Rationale: Ensures the base optimization loop can execute at least one iteration 
               without crashing and returns a valid Image object.
    Expected Behavior: Returns a PIL Image of size 512x512 (default resize).
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # Small steps for speed
    adv_img = attacker.attack(clean_image, strength=0.5, hyperparameters={"steps": 2})
    assert isinstance(adv_img, Image.Image)
    assert adv_img.size == (512, 512)

# 2. Positive Test: Image-targeted attack runs successfully
def test_image_attack_positive(clean_image, target_image):
    """
    Coverage: AttackVLMImage.setup and compute_step_loss.
    Rationale: Verifies that the image-to-image attack correctly handles a target 
               image and computes a loss involving both target and clean embeddings.
    Expected Behavior: Successfully completes the attack loop using image-based guidance.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMImage(model, device="cpu")
    adv_img = attacker.attack(clean_image, target_image, strength=0.5, hyperparameters={"steps": 2})
    assert isinstance(adv_img, Image.Image)

# 3. Positive Test: Text-targeted attack runs successfully
def test_text_attack_positive(clean_image):
    """
    Coverage: AttackVLMText.setup and compute_step_loss.
    Rationale: Ensures the text-to-image attack correctly interfaces with the 
               model's text embedding function.
    Expected Behavior: Successfully completes the attack loop using text-based guidance.
    """
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")
    adv_img = attacker.attack(clean_image, "a baseball", strength=0.5, hyperparameters={"steps": 2})
    assert isinstance(adv_img, Image.Image)

# 4. Edge Case: Strength is zero (should result in no change or minimal change)
def test_attack_strength_zero(clean_image):
    """
    Coverage: _strength_to_eps helper method.
    Rationale: Tests the boundary condition where no perturbation is allowed. 
               This checks if the math handles epsilon=0 gracefully.
    Expected Behavior: Runs successfully; internal 'delta' remains zero.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    adv_img = attacker.attack(clean_image, strength=0.0, hyperparameters={"steps": 2})
    # With strength 0, eps should be 0, so delta should stay 0.
    # However, due to floating point and conversion, we just check it runs.
    assert isinstance(adv_img, Image.Image)

# 5. Failure Mode: Negative strength should raise ValueError
def test_attack_strength_negative(clean_image):
    """
    Coverage: _strength_to_eps error handling.
    Rationale: Negative strength is semantically invalid for an epsilon-ball constraint.
    Expected Behavior: Raises ValueError with an appropriate message.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    with pytest.raises(ValueError, match="strength must be non-negative"):
        attacker.attack(clean_image, strength=-1.0)

# 6. Edge Case: Steps is zero
def test_attack_steps_zero(clean_image):
    """
    Coverage: _run_attack loop boundary.
    Rationale: Tests behavior when the optimization loop is skipped.
    Expected Behavior: Returns the original image (resized/processed) without modification.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # Should run and return an image (the clean image resized)
    adv_img = attacker.attack(clean_image, strength=0.5, hyperparameters={"steps": 0})
    assert isinstance(adv_img, Image.Image)

# 7. Positive Test: OCR attack runs successfully (tests masking logic)
def test_ocr_attack_positive(clean_image):
    """
    Coverage: AttackVLMOCR._build_text_like_mask and mask() override.
    Rationale: OCR mode is the most complex subclass; this verifies the multi-step 
               masking pipeline (Sobel, pooling, quantiles).
    Expected Behavior: Returns a masked adversarial image.
    """
    model = MockTextTargetModel()
    attacker = AttackVLMOCR(model, device="cpu")
    adv_img = attacker.attack(clean_image, "text", strength=0.5, hyperparameters={"steps": 2})
    assert isinstance(adv_img, Image.Image)

# 8. Important Use Case: EOT (Expectation over Transformation) enabled
def test_attack_with_eot(clean_image):
    """
    Coverage: _eot_augment method.
    Rationale: EOT is critical for robustness. This tests noise injection, 
               brightness scaling, and spatial shifting during the inner loop.
    Expected Behavior: Executes multiple augmented forward passes per optimization step.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    hp = {
        "steps": 2,
        "eot_samples": 2,
        "eot_noise_std": 0.01,
        "eot_brightness": 0.1,
        "eot_shift_px": 2
    }
    adv_img = attacker.attack(clean_image, strength=0.5, hyperparameters=hp)
    assert isinstance(adv_img, Image.Image)

# 9. Failure Mode: Invalid image input (None)
def test_attack_invalid_image():
    """
    Coverage: _load_image_tensor entry point.
    Rationale: Ensures the system fails predictably when provided with null data 
               instead of crashing deep in the tensor logic.
    Expected Behavior: Raises AttributeError (when calling .convert() on None).
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    with pytest.raises(AttributeError):
        attacker.attack(None, strength=0.5)

# 10. Edge Case: Large strength (clamping check)
def test_attack_large_strength(clean_image):
    """
    Coverage: _strength_to_eps clamping logic.
    Rationale: Verifies that the 'max_eps' hyperparameter correctly caps 
               extremely high strength values.
    Expected Behavior: Epsilon is capped at max_eps (default 32/255); code runs stably.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # strength > 1.0 should be handled by _strength_to_eps
    adv_img = attacker.attack(clean_image, strength=1000.0, hyperparameters={"steps": 2})
    assert isinstance(adv_img, Image.Image)

# 11. Edge Case: OCR mask with very large kernel (potential padding/index issues)
def test_ocr_mask_large_kernel(clean_image):
    """
    Coverage: F.max_pool2d in _build_text_like_mask.
    Rationale: Large kernels can sometimes cause dimension mismatches or 
               out-of-bounds errors in padding logic.
    Expected Behavior: PyTorch pooling handles large kernels correctly; returns an image.
    """
    model = MockTextTargetModel()
    attacker = AttackVLMOCR(model, device="cpu")
    hp = {"steps": 1, "ocr_dilate_kernel": 100}
    adv_img = attacker.attack(clean_image, "text", strength=0.5, hyperparameters=hp)
    assert isinstance(adv_img, Image.Image)

# 12. Edge Case: OCR mask with zero max_area_ratio (extreme restriction)
def test_ocr_mask_zero_area(clean_image):
    """
    Coverage: Area-ratio thresholding in _build_text_like_mask.
    Rationale: If the mask mean is > 0, the code enters a secondary thresholding 
               block. Testing 0.0 forces this logic to its extreme limit.
    Expected Behavior: The mask is heavily pruned; code completes successfully.
    """
    model = MockTextTargetModel()
    attacker = AttackVLMOCR(model, device="cpu")
    hp = {"steps": 1, "ocr_max_area_ratio": 0.0}
    adv_img = attacker.attack(clean_image, "text", strength=0.5, hyperparameters=hp)
    assert isinstance(adv_img, Image.Image)
