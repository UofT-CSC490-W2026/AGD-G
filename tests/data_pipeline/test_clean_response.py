
import pytest
import io
import torch
import sys
from unittest.mock import MagicMock, patch, ANY
from PIL import Image

# Mock sentence_transformers and transformers to avoid Keras errors and model loading
mock_st = MagicMock()
sys.modules["sentence_transformers"] = mock_st

@pytest.fixture
def mock_aws():
    with patch("agdg.data_pipeline.clean_response.get_db_connection") as mock_db, \
         patch("agdg.data_pipeline.clean_response.get_image") as mock_get_image:
        
        # Mock DB connection and cursor
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        mock_db.return_value.__enter__.return_value = conn
        
        # Mock get_image to return dummy bytes (valid PNG)
        img = Image.new("RGB", (1, 1))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        valid_img_bytes = img_byte_arr.getvalue()
        mock_get_image.return_value = valid_img_bytes
        
        yield {
            "db": mock_db,
            "conn": conn,
            "cur": cur,
            "get_image": mock_get_image,
            "valid_img_bytes": valid_img_bytes
        }

@pytest.fixture
def mock_vlm():
    with patch("transformers.AutoProcessor.from_pretrained") as mock_proc_cls, \
         patch("transformers.LlavaForConditionalGeneration.from_pretrained") as mock_model_cls, \
         patch("transformers.AutoModelForVisualQuestionAnswering.from_pretrained") as mock_auto_model_cls:
        
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "Prompt"
        
        # Mocking the BatchEncoding returned by processor()
        mock_inputs = MagicMock()
        mock_pixel_values = MagicMock()
        mock_pixel_values.to.return_value = mock_pixel_values
        
        mock_inputs.__getitem__.side_effect = lambda k: {
            "input_ids": torch.zeros((1, 5), dtype=torch.long),
            "pixel_values": mock_pixel_values
        }[k]
        mock_inputs.__contains__.side_effect = lambda k: k in ["input_ids", "pixel_values"]
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs
        
        mock_processor.batch_decode.return_value = ["A bar chart showing data."]
        mock_proc_cls.return_value = mock_processor
        
        # Mock LLaVA
        mock_llava = MagicMock()
        mock_llava.generate.return_value = torch.zeros((1, 10), dtype=torch.long)
        mock_llava.to.return_value = mock_llava
        mock_model_cls.return_value = mock_llava
        
        # Mock AutoModel
        mock_auto = MagicMock()
        mock_auto.generate.return_value = torch.zeros((1, 10), dtype=torch.long)
        mock_auto.to.return_value = mock_auto
        mock_auto_model_cls.return_value = mock_auto
        
        yield {
            "processor": mock_processor,
            "llava": mock_llava,
            "auto": mock_auto,
            "inputs": mock_inputs,
            "pixel_values": mock_pixel_values
        }

def test_generate_clean_responses_no_rows(mock_aws, mock_vlm):
    from agdg.data_pipeline.clean_response import generate_clean_responses
    mock_aws["cur"].fetchall.return_value = []
    
    result = generate_clean_responses()
    assert result == {"processed": 0}
    mock_aws["cur"].execute.assert_called_with(ANY)

def test_generate_clean_responses_llava_success(mock_aws, mock_vlm):
    from agdg.data_pipeline.clean_response import generate_clean_responses
    mock_aws["cur"].fetchall.return_value = [(1, "uuid-1")]
    
    result = generate_clean_responses(model_id="llava-1.5")
    
    assert result == {"processed": 1}
    assert mock_vlm["llava"].generate.called
    assert mock_vlm["processor"].apply_chat_template.called

def test_generate_clean_responses_auto_model_success(mock_aws, mock_vlm):
    from agdg.data_pipeline.clean_response import generate_clean_responses
    mock_aws["cur"].fetchall.return_value = [(1, "uuid-1")]
    
    # Using a model name that doesn't contain 'llava'
    result = generate_clean_responses(model_id="generic-vqa-model")
    
    assert result == {"processed": 1}
    assert mock_vlm["auto"].generate.called
    assert not mock_vlm["processor"].apply_chat_template.called

def test_generate_clean_responses_with_error(mock_aws, mock_vlm):
    from agdg.data_pipeline.clean_response import generate_clean_responses
    mock_aws["cur"].fetchall.return_value = [(1, "uuid-1"), (2, "uuid-2")]
    mock_aws["get_image"].side_effect = [Exception("S3 error"), mock_aws["valid_img_bytes"]]
    
    with patch("agdg.data_pipeline.clean_response.BATCH_SIZE", 1):
        result = generate_clean_responses()
    
    assert result == {"processed": 1}
    assert mock_aws["conn"].commit.call_count == 1
