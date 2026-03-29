import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
try:
    from core.attackvlm_adapter import AttackVLMTextAdapter, AttackResult
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.attackvlm_adapter import AttackVLMTextAdapter, AttackResult

@pytest.fixture
def mock_vlm_dependencies():
    with patch("core.attackvlm_adapter.TextCLIPModel") as mock_text_clip, \
         patch("core.attackvlm_adapter.PatchTextCLIPModel") as mock_patch_clip, \
         patch("core.attackvlm_adapter.AttackVLMText") as mock_attack_vlm:
        yield {
            "text_clip": mock_text_clip,
            "patch_clip": mock_patch_clip,
            "attack_vlm": mock_attack_vlm
        }

def test_attack_result_dataclass():
    img = Image.new("RGB", (10, 10))
    res = AttackResult(adversarial=[img], success=[True], scores=[0.9])
    assert res.adversarial == [img]
    assert res.success == [True]
    assert res.scores == [0.9]

def test_adapter_initialization_patch(mock_vlm_dependencies):
    adapter = AttackVLMTextAdapter(model_name="clip_text_patch", device="cpu")
    assert mock_vlm_dependencies["patch_clip"].called
    assert not mock_vlm_dependencies["text_clip"].called
    assert mock_vlm_dependencies["attack_vlm"].called

def test_adapter_initialization_text(mock_vlm_dependencies):
    adapter = AttackVLMTextAdapter(model_name="clip_text", device="cpu")
    assert mock_vlm_dependencies["text_clip"].called
    assert not mock_vlm_dependencies["patch_clip"].called
    assert mock_vlm_dependencies["attack_vlm"].called

def test_adapter_attack_call(mock_vlm_dependencies):
    adapter = AttackVLMTextAdapter(device="cpu")
    mock_method = mock_vlm_dependencies["attack_vlm"].return_value

    clean_img = Image.new("RGB", (10, 10))
    adv_img = Image.new("RGB", (10, 10), color="red")
    mock_method.attack.return_value = [adv_img]

    result = adapter.attack([clean_img], ["target text"])

    assert len(result.adversarial) == 1
    assert result.adversarial[0] == adv_img
    assert result.success == [False] # Current implementation always returns False
    assert mock_method.attack.called

    # Verify parameters passed to the underlying attack method
    args, kwargs = mock_method.attack.call_args
    assert kwargs["clean"] == [clean_img]
    assert kwargs["target"] == ["target text"]
    assert kwargs["strength"] == adapter.strength

def test_adapter_invalid_model():
    with pytest.raises(KeyError):
        AttackVLMTextAdapter(model_name="invalid_model")
