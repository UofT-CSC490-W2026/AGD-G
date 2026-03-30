import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from tests.helpers.torch_guard import require_torch

torch = require_torch()

from agdg.attack.surrogates.llava_target import LlavaTextTargetModel

@pytest.fixture
def mock_transformers():
    with patch("transformers.AutoProcessor") as mock_processor_cls, \
         patch("transformers.LlavaForConditionalGeneration") as mock_model_cls:
        
        mock_processor = MagicMock()
        mock_processor.tokenizer.padding_side = "left"
        mock_processor.image_processor.image_mean = [0.5, 0.5, 0.5]
        mock_processor.image_processor.image_std = [0.5, 0.5, 0.5]
        mock_processor.image_processor.crop_size = {"height": 336}
        mock_processor.apply_chat_template.return_value = "Prompt: "
        
        # Mocking processor call returns
        mock_processor.return_value = {
            "input_ids": torch.zeros((1, 10), dtype=torch.long),
            "attention_mask": torch.ones((1, 10), dtype=torch.long),
        }
        
        mock_processor_cls.from_pretrained.return_value = mock_processor
        
        mock_model = MagicMock()
        mock_model.config.vision_config.patch_size = 14
        mock_model.config.vision_config.image_size = 336
        mock_model.config.vision_feature_select_strategy = "default"
        mock_model.to.return_value = mock_model
        
        # Mocking model call returns
        mock_outputs = MagicMock()
        mock_outputs.loss = torch.tensor(1.0, requires_grad=True)
        mock_model.return_value = mock_outputs
        
        mock_model_cls.from_pretrained.return_value = mock_model
        
        yield {
            "processor": mock_processor,
            "model": mock_model,
            "processor_cls": mock_processor_cls,
            "model_cls": mock_model_cls
        }

def test_llava_init(mock_transformers):
    model = LlavaTextTargetModel(device="cpu")
    assert model.device == "cpu"
    assert model.model_id == "llava-hf/llava-1.5-7b-hf"
    assert model.image_size == 336
    assert torch.allclose(model.image_mean, torch.tensor([0.48145466, 0.4578275, 0.40821073]))

def test_llava_get_image_size(mock_transformers):
    model = LlavaTextTargetModel(device="cpu")
    assert model.get_image_size() == (336, 336)

def test_llava_parse_question_answer():
    # Test default
    q, a = LlavaTextTargetModel._parse_question_answer("Just an answer")
    assert q == "What is the graph about?"
    assert a == "Just an answer"
    
    # Test explicit
    q, a = LlavaTextTargetModel._parse_question_answer("Question: What color?\nAnswer: Blue")
    assert q == "What color?"
    assert a == "Blue"

def test_llava_preprocess(mock_transformers):
    model = LlavaTextTargetModel(device="cpu")
    img = torch.zeros((1, 3, 512, 512))
    processed = model._preprocess(img)
    assert processed.shape == (1, 3, 336, 336)
    # Since mean is 0.5 and std is 0.5, 0 -> (0-0.5)/0.5 = -1.0
    assert torch.allclose(processed, torch.tensor(-1.0))

def test_llava_embed_text(mock_transformers):
    model = LlavaTextTargetModel(device="cpu")
    
    # Single text
    example = model.embed_text("Test")
    assert "input_ids" in example
    assert "labels" in example
    assert mock_transformers["processor"].apply_chat_template.called
    
    # List of text
    examples = model.embed_text(["Test 1", "Test 2"])
    assert isinstance(examples, list)
    assert len(examples) == 2

def test_llava_embed_image(mock_transformers):
    model = LlavaTextTargetModel(device="cpu")
    img = torch.randn(1, 3, 336, 336)
    
    # No detach
    out = model.embed_image(img, detach=False)
    assert out is img
    
    # Detach
    out = model.embed_image(img, detach=True)
    assert not out.requires_grad

def test_llava_call_cosine(mock_transformers):
    # Tests the fallback branch if text is a tensor
    model = LlavaTextTargetModel(device="cpu")
    img = torch.randn(1, 10)
    text = torch.randn(1, 10)
    sim = model(img, text)
    assert sim.shape == (1,)

def test_llava_call_full(mock_transformers):
    model = LlavaTextTargetModel(device="cpu")
    img = torch.randn(2, 3, 512, 512)
    example = {
        "input_ids": torch.zeros((1, 10), dtype=torch.long),
        "attention_mask": torch.ones((1, 10), dtype=torch.long),
        "labels": torch.zeros((1, 10), dtype=torch.long),
    }
    
    losses = model(img, example)
    assert losses.shape == (2,)
    # Loss in __call__ is -outputs.loss, so -1.0
    assert torch.allclose(losses, torch.tensor(-1.0))
    assert mock_transformers["model"].call_count == 2
