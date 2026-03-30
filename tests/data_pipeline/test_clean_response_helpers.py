import sys
import types
from unittest.mock import MagicMock, patch

from PIL import Image


def test_get_device_prefers_cuda_without_real_torch(monkeypatch):
    from agdg.data_pipeline.clean_response import _get_device

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _get_device() == "cuda"


def test_prepare_inputs_uses_chat_template_for_llava(monkeypatch):
    from agdg.data_pipeline.clean_response import _prepare_inputs

    class FakeInputs(dict):
        def to(self, device):
            self["device"] = device
            return self

    fake_torch = types.SimpleNamespace(float16="float16")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    processor = MagicMock()
    processor.apply_chat_template.return_value = "rendered"
    processor.return_value = FakeInputs(
        {
            "pixel_values": types.SimpleNamespace(to=lambda dtype: f"cast:{dtype}"),
            "input_ids": types.SimpleNamespace(shape=(1, 4)),
        }
    )

    inputs, rendered = _prepare_inputs(
        processor,
        "llava-model",
        Image.new("RGB", (4, 4)),
        "Question?",
        "cuda",
        "float16",
    )

    assert rendered == "rendered"
    assert inputs["device"] == "cuda"
    assert inputs["pixel_values"] == "cast:float16"
    processor.apply_chat_template.assert_called_once()


def test_prepare_inputs_uses_plain_prompt_for_generic_model(monkeypatch):
    from agdg.data_pipeline.clean_response import _prepare_inputs

    class FakeInputs(dict):
        def to(self, device):
            self["device"] = device
            return self

    fake_torch = types.SimpleNamespace(float16="float16")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    processor = MagicMock()
    processor.return_value = FakeInputs({"pixel_values": "pixels"})

    inputs, rendered = _prepare_inputs(
        processor,
        "generic-vqa",
        Image.new("RGB", (4, 4)),
        "Question?",
        "cpu",
        "float32",
    )

    assert rendered == "Question?"
    assert inputs["device"] == "cpu"
    processor.apply_chat_template.assert_not_called()


def test_generate_image_response_trims_llava_prompt(monkeypatch):
    from agdg.data_pipeline.clean_response import generate_image_response

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeOutputIds:
        def __getitem__(self, key):
            return "trimmed-output"

    fake_torch = types.SimpleNamespace(no_grad=lambda: FakeNoGrad())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    processor = MagicMock()
    processor.batch_decode.return_value = [" final answer  "]
    model = MagicMock()
    model.generate.return_value = FakeOutputIds()
    inputs = {"input_ids": types.SimpleNamespace(shape=(1, 3))}

    with patch("agdg.data_pipeline.clean_response._prepare_inputs", return_value=(inputs, "rendered")), patch(
        "PIL.Image.open",
        return_value=Image.new("RGB", (4, 4)),
    ):
        result = generate_image_response(
            processor=processor,
            model=model,
            model_id="llava-model",
            image_bytes=b"png",
            device="cpu",
            dtype="float32",
        )

    assert result == "final answer"
    model.generate.assert_called_once()


def test_load_vlm_uses_generic_vqa_path(monkeypatch):
    from agdg.data_pipeline.clean_response import load_vlm

    fake_torch = types.SimpleNamespace(float16="float16", float32="float32")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    auto_processor = MagicMock()
    generic_model = MagicMock()
    fake_transformers = types.SimpleNamespace(
        AutoProcessor=types.SimpleNamespace(from_pretrained=MagicMock(return_value=auto_processor)),
        AutoModelForVisualQuestionAnswering=types.SimpleNamespace(
            from_pretrained=MagicMock(return_value=generic_model)
        ),
        LlavaForConditionalGeneration=types.SimpleNamespace(from_pretrained=MagicMock()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with patch("agdg.data_pipeline.clean_response._get_device", return_value="cpu"):
        processor, model, device, dtype = load_vlm("generic-vqa")

    assert processor is auto_processor
    assert model is generic_model.to.return_value
    assert device == "cpu"
    assert dtype == "float32"
    fake_transformers.AutoModelForVisualQuestionAnswering.from_pretrained.assert_called_once()
    fake_transformers.LlavaForConditionalGeneration.from_pretrained.assert_not_called()
