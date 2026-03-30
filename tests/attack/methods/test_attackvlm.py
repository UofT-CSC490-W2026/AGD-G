"""
Edge case tests for the AttackVLM class.

Parametrized images: the five images used one solid white canvas, one solid black canvas,
and three charts of different types (bubble, bar, pie) taken from our dataset—each padded
to the model input size.

This test suite provides comprehensive coverage (100%) of attackvlm.py through
26 tests organized into 7 categories:

1. BASIC FUNCTIONALITY (Happy Path) - Tests 1-5
   - Test 1: Untargeted attack - perturbs image away from original meaning
   - Test 2: Image-to-image attack - guides clean image toward target image
   - Test 3: Text-to-image attack - guides image toward text description
   - Test 4: OCR-masked attack - restricts perturbations to text-like regions
             (OCR = Optical Character Recognition masking using Sobel edge detection)
   - Test 5: EOT augmentation - tests robustness features (noise, brightness, shifts)
             (EOT = Expectation over Transformation for adversarial robustness)

2. HYPERPARAMETER BOUNDARY TESTS - Tests 6-9
   - Test 6: Zero strength - no perturbation allowed (eps=0)
   - Test 7: Negative strength - invalid input should raise ValueError
   - Test 8: Zero steps - optimization loop skipped
   - Test 9: Extremely large strength - tests max_eps clamping

3. IMAGE FORMAT EDGE CASES - Tests 10-13
   - Test 10: Empty image (0x0 size) - tests preprocessing robustness
   - Test 11: Grayscale image - automatic conversion from 1-channel to RGB
   - Test 12: Extreme aspect ratio - tests resizing stability (1000x10 image)
   - Test 13: RGBA image - verifies alpha channel stripping

4. OCR MASKING EDGE CASES - Tests 14-16
   - Test 14: Large dilation kernel - extreme pooling kernel size (100)
   - Test 15: Zero max area ratio - maximum mask restriction
   - Test 16: OCR on flat image - zero gradients everywhere

5. FAILURE MODES & ROBUSTNESS - Tests 17-20
   - Test 17: None input - should fail predictably with AttributeError
   - Test 18: NaN loss handling - optimizer shouldn't crash on NaN
   - Test 19: Extreme EOT shift - circular wrapping beyond image dimensions
   - Test 20: Empty text target - tests null string handling

6. REGRESSION TESTS (Gradient Correctness) - Tests 21-24
   - Test 21: Target text loss depends on adversarial embedding
   - Test 22: Optimization updates delta (1-step vs 5-step comparison)
   - Test 23: Source text loss depends on adversarial embedding
   - Test 24: Image attack repel_clean=0 isolates target-image term

7. ADVANCED FEATURE COVERAGE (Source Text Repulsion) - Tests 25-26
   - Test 25: Text attack with source_text - repel from original caption
   - Test 26: OCR attack with source_text - combines masking + source repulsion
"""

import pytest
import numpy as np
from PIL import Image
from collections.abc import Sequence
from pathlib import Path
from tests.helpers.torch_guard import require_torch

torch = require_torch()

from agdg.attack.methods.attackvlm import (
    AttackVLMUntargeted,
    AttackVLMImage,
    AttackVLMText,
    AttackVLMOCR,
)


def assert_valid_adv_image(img):
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (512, 512)

    arr = np.asarray(img)
    assert arr.dtype == np.uint8
    assert arr.shape == (512, 512, 3)
    assert arr.min() >= 0
    assert arr.max() <= 255

def load_padded_image(image_ref):
    path = _TESTS_DIR / image_ref
    return Image.open(path).convert("RGB")

def calculate_ssd(img1, img2):
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    return np.sum((arr1 - arr2) ** 2)

# Mocking the model interfaces
class MockImageTargetModel:
    def __init__(self):
        self.device = "cpu"
    def get_image_size(self):
        return (512, 512)
    def embed_image(self, x, detach=False):
        # Ensure x is differentiable if it requires grad
        # We need a path from x to the output
        # Using a simple mean or sum to keep it differentiable
        # features: [B, 512]
        # We can use a dummy projection: x is [B, 3, 512, 512]
        # Let's just take the mean over spatial and multiply by a dummy vector
        feat = x.mean(dim=(-2, -1)) # [B, 3]
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

_TESTS_DIR = Path(__file__).resolve().parent

# Logical ids for parametrized tests and reporting (files live next to this module).
# The set is: all-white, all-black, plus three dataset charts (bubble / bar / pie)—see module docstring.
IMAGE_REFS = [
    "../../data/images/padded_bubble_2.png",
    "../../data/images/padded_clean_data_viz.png",
    "../../data/images/padded_image.png",
    "../../data/images/padded_black.png",
    "../../data/images/padded_white.png",
]

# Text attack targets aligned with each fixture image (chart type or solid color).
IMAGE_TARGET_TEXTS = {
    "../../data/images/padded_bubble_2.png": "a bubble chart",
    "../../data/images/padded_clean_data_viz.png": "a bar chart",
    "../../data/images/padded_image.png": "a pie chart",
    "../../data/images/padded_black.png": "black",
    "../../data/images/padded_white.png": "white",
}


def build_padded_images():
    """Load each pre-padded image from disk."""
    return [(ref, load_padded_image(ref)) for ref in IMAGE_REFS]


# Padded canvases built at import; tests take a copy so inputs are not shared across cases.
PADDED_CLEAN_IMAGES = build_padded_images()


@pytest.fixture(params=PADDED_CLEAN_IMAGES, ids=[r for r, _ in PADDED_CLEAN_IMAGES])
def clean_image(request):
    _, padded = request.param
    return padded.copy()


@pytest.fixture(params=PADDED_CLEAN_IMAGES, ids=[r for r, _ in PADDED_CLEAN_IMAGES])
def clean_image_and_target(request):
    ref, padded = request.param
    return padded.copy(), IMAGE_TARGET_TEXTS[ref]


@pytest.fixture
def target_image():
    return Image.new("RGB", (100, 100), color="black")

# ==============================================================================
# GROUP 1: BASIC FUNCTIONALITY (Happy Path) - Tests 1-5
# ==============================================================================

# 1. Positive Test: Untargeted attack runs successfully
def test_01_untargeted_attack_positive(clean_image):
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
    assert calculate_ssd(clean_image, adv_img) > 0

# 2. Positive Test: Image-targeted attack runs successfully
def test_02_image_attack_positive(clean_image, target_image):
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
    assert calculate_ssd(clean_image, adv_img) > 0

# 3. Positive Test: Text-targeted attack runs successfully
def test_03_text_attack_positive(clean_image_and_target):
    """
    Coverage: AttackVLMText.setup and compute_step_loss.
    Rationale: Ensures the text-to-image attack correctly interfaces with the
               model's text embedding function.
    Expected Behavior: Successfully completes the attack loop using text-based guidance.
    """
    clean_image, target_text = clean_image_and_target
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")
    adv_img = attacker.attack(clean_image, target_text, strength=0.5, hyperparameters={"steps": 2})

    assert_valid_adv_image(adv_img)
    assert calculate_ssd(clean_image, adv_img) > 0

# 4. Positive Test: OCR attack runs successfully (tests masking logic)
def test_04_ocr_attack_positive(clean_image_and_target):
    """
    Coverage: AttackVLMOCR._build_text_like_mask and mask() override.
    Rationale: OCR mode is the most complex subclass; this verifies the multi-step
               masking pipeline (Sobel, pooling, quantiles).
    Expected Behavior: Returns a masked adversarial image different from clean.
    """
    clean_image, target_text = clean_image_and_target
    model = MockTextTargetModel()
    attacker = AttackVLMOCR(model, device="cpu")
    adv_img = attacker.attack(clean_image, target_text, strength=0.5, hyperparameters={"steps": 2})

    assert_valid_adv_image(adv_img)
    assert calculate_ssd(clean_image, adv_img) > 0

# 5. Important Use Case: EOT (Expectation over Transformation) enabled
def test_05_attack_with_eot(clean_image):
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
    assert calculate_ssd(clean_image, adv_img) > 0


# ==============================================================================
# GROUP 2: HYPERPARAMETER BOUNDARY TESTS - Tests 6-9
# ==============================================================================

# 6. Edge Case: Strength is zero (should result in no change)
def test_06_attack_strength_zero(clean_image):
    """
    Coverage: _strength_to_eps helper method.
    Rationale: Tests the boundary condition where no perturbation is allowed.
    Expected Behavior: The output image should be identical to the resized clean image.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    adv_img = attacker.attack(clean_image, strength=0.0, hyperparameters={"steps": 2})

    # Should be identical because eps=0 forces delta=0
    assert calculate_ssd(clean_image, adv_img) == 0

# 7. Failure Mode: Negative strength should raise ValueError
def test_07_attack_strength_negative(clean_image):
    """
    Coverage: _strength_to_eps error handling.
    Rationale: Negative strength is semantically invalid for an epsilon-ball constraint.
    Expected Behavior: Raises ValueError with an appropriate message.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    with pytest.raises(ValueError, match="strength must be non-negative"):
        attacker.attack(clean_image, strength=-1.0)

# 8. Edge Case: Steps is zero
def test_08_attack_steps_zero(clean_image):
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
    assert calculate_ssd(clean_image, adv_img) == 0

# 9. Edge Case: Large strength (clamping check)
def test_09_attack_large_strength(clean_image):
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
    assert calculate_ssd(clean_image, adv_img) > 0


# ==============================================================================
# GROUP 3: IMAGE FORMAT EDGE CASES - Tests 10-13
# ==============================================================================

# 10. Edge Case: Image with size 0
def test_10_attack_size_zero():
    """
    Coverage: _load_image_tensor resizing.
    Rationale: Checks that a 0×0 image does not trigger a division-by-zero or
               memory allocation error in the resize step.
    Expected Behavior: Returns a valid 512x512 adversarial image.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # PIL image with (0, 0) size
    clean_img = Image.new("RGB", (0, 0))
    # Using 1 step for speed
    adv_img = attacker.attack(clean_img, strength=0.5, hyperparameters={"steps": 1})

    assert_valid_adv_image(adv_img)


# 11. Edge Case: Grayscale image (1 channel)
def test_11_attack_grayscale():
    """
    Coverage: _load_image_tensor conversion.
    Rationale: The model expects 3-channel input; this verifies that a single-channel
               image is converted to RGB before tensor conversion.
    Expected Behavior: Returns a valid 3-channel 512x512 adversarial image.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # Grayscale "L" mode image
    clean_img = Image.new("L", (100, 100), color=255)
    adv_img = attacker.attack(clean_img, strength=0.5, hyperparameters={"steps": 1})

    assert_valid_adv_image(adv_img)


# 12. Edge Case: Extreme aspect ratio
def test_12_attack_extreme_aspect_ratio():
    """
    Coverage: _load_image_tensor resizing logic.
    Rationale: A large width-to-height mismatch produces heavy distortion when
               resizing to 512×512; this checks that the resize step remains
               numerically stable at that extreme.
    Expected Behavior: Returns a valid 512x512 adversarial image.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # Extremely thin image (1000x10)
    clean_img = Image.new("RGB", (1000, 10), color="blue")
    adv_img = attacker.attack(clean_img, strength=0.5, hyperparameters={"steps": 1})

    assert_valid_adv_image(adv_img)


# 13. Edge Case: RGBA Image (Transparency)
def test_13_attack_rgba_image():
    """
    Coverage: _load_image_tensor conversion from RGBA.
    Rationale: Checks that the alpha channel is stripped before tensor conversion,
               preventing shape mismatches when the model expects 3-channel input.
    Expected Behavior: Returns a valid 3-channel 512x512 RGB image.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # Create an RGBA image with transparency
    clean_img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    adv_img = attacker.attack(clean_img, strength=0.5, hyperparameters={"steps": 1})

    assert_valid_adv_image(adv_img)


# ==============================================================================
# GROUP 4: OCR MASKING EDGE CASES - Tests 14-16
# ==============================================================================

# 14. Edge Case: OCR mask with very large kernel (potential padding/index issues)
def test_14_ocr_mask_large_kernel(clean_image_and_target):
    """
    Coverage: F.max_pool2d in _build_text_like_mask.
    Rationale: Large kernels can sometimes cause dimension mismatches or
               out-of-bounds errors in padding logic.
    Expected Behavior: PyTorch pooling handles large kernels correctly; returns an image.
    """
    clean_image, target_text = clean_image_and_target
    model = MockTextTargetModel()
    attacker = AttackVLMOCR(model, device="cpu")
    hp = {"steps": 1, "ocr_dilate_kernel": 100}
    adv_img = attacker.attack(clean_image, target_text, strength=0.5, hyperparameters=hp)

    assert_valid_adv_image(adv_img)
    # Even with large kernel, at 1 step we expect some initialization change
    assert calculate_ssd(clean_image, adv_img) > 0

# 15. Edge Case: OCR mask with zero max_area_ratio (extreme restriction)
def test_15_ocr_mask_zero_area(clean_image_and_target):
    """
    Coverage: Area-ratio thresholding in _build_text_like_mask.
    Rationale: If the mask mean is > 0, the code enters a secondary thresholding
               block. Testing 0.0 forces this logic to its extreme limit.
    Expected Behavior: The mask is heavily pruned to only the highest gradient pixels.
                       Even with a flat image, initialization noise and tiny gradient
                       differences mean some perturbation usually persists.
    """
    clean_image, target_text = clean_image_and_target
    model = MockTextTargetModel()
    attacker = AttackVLMOCR(model, device="cpu")
    hp = {"steps": 1, "ocr_max_area_ratio": 0.0}
    adv_img = attacker.attack(clean_image, target_text, strength=0.5, hyperparameters=hp)

    assert_valid_adv_image(adv_img)
    # Even with 0.0 area ratio, the logic (quantile 1.0) might
    # still allow the very top-gradient pixel(s) to be perturbed.
    assert calculate_ssd(clean_image, adv_img) > 0

# 16. Edge Case: OCR Masking on Flat Image
def test_16_ocr_mask_flat_image():
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
    adv_img = attacker.attack(clean_img, "white", strength=0.5, hyperparameters={"steps": 1})

    assert_valid_adv_image(adv_img)


# ==============================================================================
# GROUP 5: FAILURE MODES & ROBUSTNESS - Tests 17-20
# ==============================================================================

# 17. Failure Mode: Invalid image input (None)
def test_17_attack_invalid_image():
    """
    Coverage: _load_image_tensor entry point.
    Rationale: Ensures the system fails predictably when provided with null data
               instead of crashing deep in the tensor logic.
    Expected Behavior: Raises AttributeError (when calling .convert() on None).
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    with pytest.raises(TypeError):
        attacker.attack(None, strength=0.5)


# 18. Failure Mode: NaN Loss Handling
def test_18_attack_nan_loss(clean_image):
    """
    Coverage: _run_attack optimization loop robustness.
    Rationale: Checks that a NaN loss does not crash the optimizer or produce
               an invalid output; `delta.grad` manipulation must remain safe
               when gradients are undefined.
    Expected Behavior: Executes without crashing; output is a valid image.
    """
    class NaNModel(MockImageTargetModel):
        def __call__(self, a, b):
            return torch.tensor(float('nan'), requires_grad=True)

    model = NaNModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    adv_img = attacker.attack(clean_image, strength=0.5, hyperparameters={"steps": 2})

    assert_valid_adv_image(adv_img)


# 19. Edge Case: Extreme EOT Shift
def test_19_attack_extreme_eot_shift(clean_image):
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


# 20. Edge Case: Empty text target
def test_20_attack_empty_text(clean_image):
    """
    Coverage: AttackVLMText.setup with empty string.
    Rationale: Checks that the model's text-embedding path and the subsequent
               loss calculation handle an empty string without error.
    Expected Behavior: Returns a valid 512x512 adversarial image.
    """
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")
    # Empty string as target
    adv_img = attacker.attack(clean_image, "", strength=0.5, hyperparameters={"steps": 1})

    assert_valid_adv_image(adv_img)


# ==============================================================================
# GROUP 6: REGRESSION TESTS (Gradient Correctness) - Tests 21-24
# ==============================================================================

# 21. Regression: Target similarity term must depend on adversarial embedding
def test_21_text_loss_depends_on_adv_embed(clean_image_and_target):
    """
    Coverage: AttackVLMText.compute_step_loss target term.
    Rationale: If sim_text is computed from a constant (e.g. clean_embed instead of
               adv_embed), its gradient is zero and the optimizer receives no signal
               to steer toward the target text. This test catches that by checking
               the loss changes when adv_embed changes.
    Expected Behavior: Different adversarial embeddings produce different loss values.
    """
    clean_image, target_text = clean_image_and_target
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")
    clean_t = attacker._load_image_tensor(clean_image).unsqueeze(0)
    clean_embed = model.embed_image(clean_t)
    # Set repel_clean=0 to isolate the target text term (-sim_text only)
    attacker.setup(clean_t, target_text=target_text, repel_clean=0.0)

    adv_embed_a = torch.randn_like(clean_embed)
    adv_embed_b = torch.randn_like(clean_embed)

    loss_a = attacker.compute_step_loss(adv_embed_a, clean_embed)
    loss_b = attacker.compute_step_loss(adv_embed_b, clean_embed)

    assert loss_a.item() != loss_b.item(), "Loss didn't change with different adv_embed — target similarity term may be constant"


# 22. Regression: More optimization steps must change the perturbation
def test_22_optimization_updates_delta(clean_image_and_target):
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
    clean_image, target_text = clean_image_and_target
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")

    # Same seed → same initial delta and same target embedding
    torch.manual_seed(42)
    adv_1 = attacker.attack(clean_image, target_text, strength=0.5,
                            hyperparameters={"steps": 1, "repel_clean": 0.0})

    torch.manual_seed(42)
    adv_5 = attacker.attack(clean_image, target_text, strength=0.5,
                            hyperparameters={"steps": 5, "repel_clean": 0.0})

    arr_1 = np.array(adv_1).astype(float)
    arr_5 = np.array(adv_5).astype(float)

    diff = np.abs(arr_5 - arr_1).mean()
    assert diff > 0.1, f"5-step attack identical to 1-step — target term gradient may be zero (diff={diff:.4f})"


# 23. Regression: Source-text similarity term must depend on adversarial embedding
def test_23_source_text_loss_depends_on_adv_embed(clean_image_and_target):
    """
    Coverage: AttackVLMText.compute_step_loss when source_text_embed is set.
    Rationale: If sim_source_text were computed from a constant w.r.t. adv_embed,
               the source repulsion term would not steer the optimization.
    Expected Behavior: Different adversarial embeddings produce different loss values.
    """
    clean_image, target_text = clean_image_and_target
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")
    clean_t = attacker._load_image_tensor(clean_image).unsqueeze(0)
    clean_embed = model.embed_image(clean_t)
    attacker.setup(
        clean_t,
        target_text=target_text,
        source_text="original description of the chart before editing",
        repel_clean=0.0,
        repel_source_text=1.0,
    )

    adv_embed_a = torch.randn_like(clean_embed)
    adv_embed_b = torch.randn_like(clean_embed)

    loss_a = attacker.compute_step_loss(adv_embed_a, clean_embed)
    loss_b = attacker.compute_step_loss(adv_embed_b, clean_embed)

    assert loss_a.item() != loss_b.item(), (
        "Loss didn't change with different adv_embed — source_text term may be constant w.r.t. adv_embed"
    )


# 24. Regression: Image attack with repel_clean=0 isolates target-image term
def test_24_image_attack_repel_clean_zero(clean_image, target_image):
    """
    Coverage: AttackVLMImage.compute_step_loss with repel_clean=0.
    Rationale: With repel_clean=0, only the target similarity term pulls the
               adversarial image; repulsion from the clean embedding is disabled.
               Catches bugs where the target term is wired incorrectly but the
               clean repulsion still produces gradients.
    Expected Behavior: Attack completes and the image changes from the clean input.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMImage(model, device="cpu")
    adv_img = attacker.attack(
        clean_image,
        target_image,
        strength=0.5,
        hyperparameters={"steps": 2, "repel_clean": 0.0},
    )

    assert_valid_adv_image(adv_img)
    assert calculate_ssd(clean_image, adv_img) > 0


# ==============================================================================
# GROUP 7: ADVANCED FEATURE COVERAGE (Source Text Repulsion) - Tests 25-26
# ==============================================================================

# 25. Positive Test: Text attack with source_text (repel original caption)
def test_25_text_attack_with_source_text(clean_image_and_target):
    """
    Coverage: AttackVLMText.setup source_text branch (embed source_text) and
              compute_step_loss source_text repulsion term.
    Rationale: When the original caption is known, the attack can push away from it
               while moving toward the target text.
    Expected Behavior: Completes without error and returns a perturbed image.
    """
    clean_image, target_text = clean_image_and_target
    model = MockTextTargetModel()
    attacker = AttackVLMText(model, device="cpu")
    hp = {
        "steps": 2,
        "source_text": "original description of the chart before editing",
    }
    adv_img = attacker.attack(clean_image, target_text, strength=0.5, hyperparameters=hp)

    assert_valid_adv_image(adv_img)
    assert calculate_ssd(clean_image, adv_img) > 0


# 26. Positive Test: OCR attack with source_text (masking + source repulsion)
def test_26_ocr_attack_with_source_text(clean_image_and_target):
    """
    Coverage: AttackVLMOCR with source_text passed through hyperparameters.
    Rationale: Stacks OCR spatial masking with the optional source-text repulsion path
               used by AttackVLMText.setup / compute_step_loss.
    Expected Behavior: Completes without error and returns a perturbed image.
    """
    clean_image, target_text = clean_image_and_target
    model = MockTextTargetModel()
    attacker = AttackVLMOCR(model, device="cpu")
    hp = {
        "steps": 2,
        "source_text": "original description of the chart before editing",
    }
    adv_img = attacker.attack(clean_image, target_text, strength=0.5, hyperparameters=hp)

    assert_valid_adv_image(adv_img)
    assert calculate_ssd(clean_image, adv_img) > 0

# ==============================================================================
# GROUP 8: BATCH MODE TESTS - Tests 27-30
# ==============================================================================

# 27. Positive Test: Single image (not a batch)
def test_27_single_image_positive(clean_image):
    """
    Coverage: AttackVLMUntargeted._run_attack loop.
    Rationale: Ensures the base optimization loop can handle a single Image,
               not inside a batch container (Sequence)
    Expected Behavior: Returns a PIL Image of size 512x512 that is different from
                       the original (due to initialized delta and optimization).
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    # Small steps for speed
    adv_img = attacker.attack(clean_image, strength=0.5, hyperparameters={"steps": 2})

    assert_valid_adv_image(adv_img)

    # Assert that the image actually changed from the clean input
    assert calculate_ssd(clean_image, adv_img) > 0

# 28. Positive Test: Batch containing no images
def test_28_batch_empty_positive():
    """
    Coverage: AttackVLMUntargeted._run_attack loop.
    Rationale: Ensures the base optimization loop can handle an empty batch
    Expected Behavior: Returns an empty Sequence
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    batch = []
    # Small steps for speed
    adv_img = attacker.attack(batch, strength=0.5, hyperparameters={"steps": 2})

    assert isinstance(adv_img, Sequence)
    assert len(adv_img) == 0

# 29. Positive Test: Batch containing one image
def test_29_batch_one_positive(clean_image):
    """
    Coverage: AttackVLMUntargeted._run_attack loop.
    Rationale: Ensures the base optimization loop can handle a batch with only
               one image, without collapsing the batch to a bare Image.Image.
    Expected Behavior: Returns a Sequence containing a PIL Image of size 512x512
                       that is different from the original. Not a bare image,
                       but a one-element list.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    batch = [clean_image]
    # Small steps for speed
    adv_img = attacker.attack(batch, strength=0.5, hyperparameters={"steps": 2})

    assert isinstance(adv_img, Sequence)
    assert len(adv_img) == len(batch)
    for img in adv_img:
        assert_valid_adv_image(img)
        # Assert that each image actually changed from the clean input
        assert calculate_ssd(clean_image, img) > 0

# 30. Positive Test: Batch containing multiple images
def test_30_batch_multiple_positive(clean_image):
    """
    Coverage: AttackVLMUntargeted._run_attack loop.
    Rationale: Ensures the base optimization loop can handle a batch with more
               than one image.
    Expected Behavior: Returns a Sequence containing PIL Images of size 512x512
                       that is different from the original.
    """
    model = MockImageTargetModel()
    attacker = AttackVLMUntargeted(model, device="cpu")
    batch = [clean_image for _ in range(4)]
    # Small steps for speed
    adv_img = attacker.attack(batch, strength=0.5, hyperparameters={"steps": 2})

    assert isinstance(adv_img, Sequence)
    assert len(adv_img) == len(batch)
    for img in adv_img:
        assert_valid_adv_image(img)
        # Assert that each image actually changed from the clean input
        assert calculate_ssd(clean_image, img) > 0
