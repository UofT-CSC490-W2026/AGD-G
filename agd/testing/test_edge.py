"""
Edge case tests for the AttackVLM class.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from agd.core.attacks.attackvlm import (
    AttackVLMUntargeted,
    AttackVLMImage,
    AttackVLMText,
    AttackVLMOCR,
)
import torch.nn.functional as F
from typing import Optional, Dict
from tqdm import tqdm


def assert_valid_adv_image(img):
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (512, 512)

    arr = np.asarray(img)
    assert arr.dtype == np.uint8
    assert arr.shape == (512, 512, 3)
    assert arr.min() >= 0
    assert arr.max() <= 255

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
    Expected Behavior: Returns a PIL Image of size 512x512 that is different from 
                       the original (due to initialized delta and optimization).
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # Small steps for speed
    adv_img = attacker.attack(clean_image, strength=0.5, hyperparameters={"steps": 2})
    
    assert_valid_adv_image(adv_img)
    
    # Assert that the image actually changed from the clean input
    clean_resized = clean_image.convert("RGB").resize((512, 512))
    assert np.array(adv_img).tolist() != np.array(clean_resized).tolist()

# 2. Positive Test: Image-targeted attack runs successfully
def test_image_attack_positive(clean_image, target_image):
    """
    Coverage: AttackVLMImage.setup and compute_step_loss.
    Rationale: Verifies that the image-to-image attack correctly handles a target 
               image and computes a loss involving both target and clean embeddings.
    Expected Behavior: Successfully completes the attack loop and produces a 
                       perturbed image.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMImage(model, device="cpu")
    adv_img = attacker.attack(clean_image, target_image, strength=0.5, hyperparameters={"steps": 2})
    
    assert_valid_adv_image(adv_img)
    clean_resized = clean_image.convert("RGB").resize((512, 512))
    assert np.array(adv_img).tolist() != np.array(clean_resized).tolist()

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
    
    assert_valid_adv_image(adv_img)
    clean_resized = clean_image.convert("RGB").resize((512, 512))
    assert np.array(adv_img).tolist() != np.array(clean_resized).tolist()

# 4. Edge Case: Strength is zero (should result in no change)
def test_attack_strength_zero(clean_image):
    """
    Coverage: _strength_to_eps helper method.
    Rationale: Tests the boundary condition where no perturbation is allowed. 
    Expected Behavior: The output image should be identical to the resized clean image.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    adv_img = attacker.attack(clean_image, strength=0.0, hyperparameters={"steps": 2})
    
    clean_resized = clean_image.convert("RGB").resize((512, 512))
    # Should be identical because eps=0 forces delta=0
    assert np.array(adv_img).tolist() == np.array(clean_resized).tolist()

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
    Expected Behavior: Since the loop is skipped, best_adv remains clean_t.
                       The output image should be identical to the resized clean image.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    adv_img = attacker.attack(clean_image, strength=0.5, hyperparameters={"steps": 0})
    
    assert_valid_adv_image(adv_img)
    clean_resized = clean_image.convert("RGB").resize((512, 512))
    assert np.array(adv_img).tolist() == np.array(clean_resized).tolist()

# 7. Positive Test: OCR attack runs successfully (tests masking logic)
def test_ocr_attack_positive(clean_image):
    """
    Coverage: AttackVLMOCR._build_text_like_mask and mask() override.
    Rationale: OCR mode is the most complex subclass; this verifies the multi-step 
               masking pipeline (Sobel, pooling, quantiles).
    Expected Behavior: Returns a masked adversarial image different from clean.
    """
    model = MockTextTargetModel()
    attacker = AttackVLMOCR(model, device="cpu")
    adv_img = attacker.attack(clean_image, "text", strength=0.5, hyperparameters={"steps": 2})
    
    assert_valid_adv_image(adv_img)
    clean_resized = clean_image.convert("RGB").resize((512, 512))
    assert np.array(adv_img).tolist() != np.array(clean_resized).tolist()

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
    
    assert_valid_adv_image(adv_img)
    clean_resized = clean_image.convert("RGB").resize((512, 512))
    assert np.array(adv_img).tolist() != np.array(clean_resized).tolist()

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
    
    assert_valid_adv_image(adv_img)
    clean_resized = clean_image.convert("RGB").resize((512, 512))
    assert np.array(adv_img).tolist() != np.array(clean_resized).tolist()

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
    
    assert_valid_adv_image(adv_img)
    clean_resized = clean_image.convert("RGB").resize((512, 512))
    # Even with large kernel, at 1 step we expect some initialization change
    assert np.array(adv_img).tolist() != np.array(clean_resized).tolist()

# 12. Edge Case: OCR mask with zero max_area_ratio (extreme restriction)
def test_ocr_mask_zero_area(clean_image):
    """
    Coverage: Area-ratio thresholding in _build_text_like_mask.
    Rationale: If the mask mean is > 0, the code enters a secondary thresholding 
               block. Testing 0.0 forces this logic to its extreme limit.
    Expected Behavior: The mask is heavily pruned to only the highest gradient pixels.
                       Even with a flat image, initialization noise and tiny gradient 
                       differences mean some perturbation usually persists.
    """
    model = MockTextTargetModel()
    attacker = AttackVLMOCR(model, device="cpu")
    hp = {"steps": 1, "ocr_max_area_ratio": 0.0}
    adv_img = attacker.attack(clean_image, "text", strength=0.5, hyperparameters=hp)
    
    assert_valid_adv_image(adv_img)
    clean_resized = clean_image.convert("RGB").resize((512, 512))
    # Even with 0.0 area ratio, the logic (quantile 1.0) might 
    # still allow the very top-gradient pixel(s) to be perturbed.
    assert np.array(adv_img).tolist() != np.array(clean_resized).tolist()


# 13. Edge Case: Image with size 0
def test_attack_size_zero():
    """
    Coverage: _load_image_tensor resizing.
    Significance: Tests the robustness of the preprocessing pipeline against empty 
                  data inputs. It ensures that a lack of pixel data doesn't trigger 
                  division-by-zero or memory allocation errors, allowing the system 
                  to gracefully handle malformed image files.
    Expected Behavior: Returns a valid 512x512 adversarial image.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # PIL image with (0, 0) size
    clean_img = Image.new("RGB", (0, 0))
    # Using 1 step for speed
    adv_img = attacker.attack(clean_img, strength=0.5, hyperparameters={"steps": 1})

    assert_valid_adv_image(adv_img)


# 14. Edge Case: Grayscale image (1 channel)
def test_attack_grayscale():
    """
    Coverage: _load_image_tensor conversion.
    Significance: Many vision-language models (like CLIP) are strictly trained on 
                  3-channel RGB data. This test ensures the library automatically 
                  bridges the gap for single-channel formats without user intervention, 
                  maintaining compatibility with standard model interfaces.
    Expected Behavior: Returns a valid 3-channel 512x512 adversarial image.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # Grayscale "L" mode image
    clean_img = Image.new("L", (100, 100), color=255)
    adv_img = attacker.attack(clean_img, strength=0.5, hyperparameters={"steps": 1})

    assert_valid_adv_image(adv_img)


# 15. Edge Case: Empty text target
def test_attack_empty_text(clean_image):
    """
    Coverage: AttackVLMText.setup with empty string.
    Significance: Tests the "boundary of meaning" by ensuring the system doesn't 
                  crash when provided with no semantic guidance. It verifies that 
                  the underlying model's embedding function and the subsequent 
                  loss calculation handle null strings gracefully.
    Expected Behavior: Returns a valid 512x512 adversarial image.
    """
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")
    # Empty string as target
    adv_img = attacker.attack(clean_image, "", strength=0.5, hyperparameters={"steps": 1})

    assert_valid_adv_image(adv_img)


# 16. Edge Case: Extreme aspect ratio
def test_attack_extreme_aspect_ratio():
    """
    Coverage: _load_image_tensor resizing logic.
    Significance: Highlights the "semantic squashing" effect of the fixed 512x512 
                  input size. It ensures the implementation remains numerically 
                  stable even when image content is heavily distorted during 
                  normalization for the target model.
    Expected Behavior: Returns a valid 512x512 adversarial image.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # Extremely thin image (1000x10)
    clean_img = Image.new("RGB", (1000, 10), color="blue")
    adv_img = attacker.attack(clean_img, strength=0.5, hyperparameters={"steps": 1})

    assert_valid_adv_image(adv_img)


# 17. Edge Case: RGBA Image (Transparency)
def test_attack_rgba_image():
    """
    Coverage: _load_image_tensor conversion from RGBA.
    Significance: Ensures that images with alpha channels (like transparent PNGs) 
                  are flattened to RGB correctly. This verifies that the 4th 
                  channel is stripped before tensor conversion, preventing 
                  shape mismatches in the model.
    Expected Behavior: Returns a valid 3-channel 512x512 RGB image.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # Create an RGBA image with transparency
    clean_img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    adv_img = attacker.attack(clean_img, strength=0.5, hyperparameters={"steps": 1})

    assert_valid_adv_image(adv_img)


# 18. Edge Case: OCR Masking on Flat Image
def test_ocr_mask_flat_image():
    """
    Coverage: AttackVLMOCR._build_text_like_mask gradient logic.
    Significance: On a solid color image, all gradients are zero. This tests 
                  if the Sobel filtering and `torch.quantile` logic handle 
                  zero-variance inputs without crashing or producing NaNs.
    Expected Behavior: Returns a valid adversarial image (mask may be zeroed out).
    """
    model = MockTextTargetModel()
    attacker = AttackVLMOCR(model, device="cpu")
    # Pure white image (zero gradients)
    clean_img = Image.new("RGB", (100, 100), color="white")
    adv_img = attacker.attack(clean_img, "text", strength=0.5, hyperparameters={"steps": 1})

    assert_valid_adv_image(adv_img)


# 19. Failure Mode: NaN Loss Handling
def test_attack_nan_loss(clean_image):
    """
    Coverage: _run_attack optimization loop robustness.
    Significance: Tests how the optimizer reacts if the model returns NaN. 
                  While the attack might fail to find a direction, it should 
                  not crash the entire process. It verifies that `delta.grad` 
                  manipulation remains safe.
    Expected Behavior: Executes without crashing; output is a valid image.
    """
    class NaNModel(MockImageTargetModel):
        def __call__(self, a, b):
            return torch.tensor(float('nan'), requires_grad=True)

    model = NaNModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    adv_img = attacker.attack(clean_image, strength=0.5, hyperparameters={"steps": 2})

    assert_valid_adv_image(adv_img)


# 20. Edge Case: Extreme EOT Shift
def test_attack_extreme_eot_shift(clean_image):
    """
    Coverage: _eot_augment with torch.roll.
    Significance: Tests if the Expectation over Transformation logic handles 
                  shift values larger than the image dimensions. `torch.roll` 
                  is circular, so this verifies that extreme values wrap around 
                  rather than causing index errors.
    Expected Behavior: Executes successfully using circular wrapping for shifts.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    hp = {"steps": 1, "eot_samples": 1, "eot_shift_px": 1000} # Larger than 512
    adv_img = attacker.attack(clean_image, strength=0.5, hyperparameters=hp)

    assert_valid_adv_image(adv_img)


# 21. Regression: Target similarity term must depend on adversarial embedding
def test_text_loss_depends_on_adv_embed(clean_image):
    """
    Coverage: AttackVLMText.compute_step_loss target term.
    Rationale: If sim_text is computed from a constant (e.g. clean_embed instead of
               adv_embed), its gradient is zero and the optimizer receives no signal
               to steer toward the target text. This test catches that by checking
               the loss changes when adv_embed changes.
    Expected Behavior: Different adversarial embeddings produce different loss values.
    """
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")
    clean_t = attacker._load_image_tensor(clean_image)
    clean_embed = model.embed_image(clean_t)
    # Set repel_clean=0 to isolate the target text term (-sim_text only)
    attacker.setup(clean_t, target_text="a baseball", repel_clean=0.0)

    adv_embed_a = torch.randn_like(clean_embed)
    adv_embed_b = torch.randn_like(clean_embed)

    loss_a = attacker.compute_step_loss(adv_embed_a, clean_embed)
    loss_b = attacker.compute_step_loss(adv_embed_b, clean_embed)

    assert loss_a.item() != loss_b.item(), "Loss didn't change with different adv_embed — target similarity term may be constant"


# 22. Regression: More optimization steps must change the perturbation
def test_optimization_updates_delta(clean_image):
    """
    Coverage: End-to-end targeted optimization signal.
    Rationale: With repel_clean=0, the only gradient signal comes from the target
               text term. If that term is accidentally constant (e.g. computed from
               clean_embed instead of adv_embed), the gradient is zero and delta
               never changes — meaning 1-step and 5-step attacks produce identical
               output. Using the same random seed ensures the initial delta is the
               same, isolating the effect of optimization.
    Expected Behavior: 5 steps of optimization should produce a different result
                       than 1 step.
    """
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")

    # Same seed → same initial delta and same target embedding
    torch.manual_seed(42)
    adv_1 = attacker.attack(clean_image, "a baseball", strength=0.5,
                            hyperparameters={"steps": 1, "repel_clean": 0.0})

    torch.manual_seed(42)
    adv_5 = attacker.attack(clean_image, "a baseball", strength=0.5,
                            hyperparameters={"steps": 5, "repel_clean": 0.0})

    arr_1 = np.array(adv_1).astype(float)
    arr_5 = np.array(adv_5).astype(float)

    diff = np.abs(arr_5 - arr_1).mean()
    assert diff > 0.1, f"5-step attack identical to 1-step — target term gradient may be zero (diff={diff:.4f})"
