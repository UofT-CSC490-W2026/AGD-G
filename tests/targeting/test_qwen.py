"""Tests for the QwenTargetingModel and helpers in qwen.py."""
import sys
import types
from unittest.mock import MagicMock

import pytest
from PIL import Image

from agdg.targeting.strategies.qwen import (
    _parse_output,
    SYSTEM_INSTRUCTION,
    FEW_SHOT_EXAMPLES,
)


class TestParseOutput:
    def test_extracts_from_tags(self):
        raw = "<thinking>some reasoning</thinking>\n<output>Bar chart of fruits</output>"
        assert _parse_output(raw) == "Bar chart of fruits"

    def test_strips_whitespace(self):
        raw = "<output>  Pie chart of toppings  </output>"
        assert _parse_output(raw) == "Pie chart of toppings"

    def test_fallback_when_no_tags(self):
        raw = "Just a plain caption"
        assert _parse_output(raw) == "Just a plain caption"

    def test_fallback_strips_whitespace(self):
        raw = "  some text  "
        assert _parse_output(raw) == "some text"

    def test_multiline_output_tag(self):
        raw = "<output>\nHeatmap of illness rates\n</output>"
        assert _parse_output(raw) == "Heatmap of illness rates"

    def test_empty_output_tag(self):
        raw = "<output></output>"
        assert _parse_output(raw) == ""


class TestGetMessage:
    @pytest.fixture()
    def model(self, monkeypatch):
        _install_fake_deps(monkeypatch)
        _clear_qwen_module()
        from agdg.targeting.strategies.qwen import QwenTargetingModel
        return QwenTargetingModel(device="cpu")

    def test_structure(self, model):
        img = Image.new("RGB", (10, 10))
        msgs = model.get_message(img, "A chart about fish")

        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"][0]["text"] == SYSTEM_INSTRUCTION

        num_few_shot = len(FEW_SHOT_EXAMPLES)
        expected_len = 1 + num_few_shot * 2 + 1  # system + pairs + user
        assert len(msgs) == expected_len

        for i, (user_text, assistant_text) in enumerate(FEW_SHOT_EXAMPLES):
            user_msg = msgs[1 + i * 2]
            asst_msg = msgs[2 + i * 2]
            assert user_msg["role"] == "user"
            assert user_msg["content"][0]["text"] == user_text
            assert asst_msg["role"] == "assistant"
            assert asst_msg["content"][0]["text"] == assistant_text

        final = msgs[-1]
        assert final["role"] == "user"
        content_types = [c["type"] for c in final["content"]]
        assert "image" in content_types
        assert "text" in content_types

    def test_final_user_contains_clean_text(self, model):
        img = Image.new("RGB", (10, 10))
        msgs = model.get_message(img, "my chart caption")
        final_texts = [c["text"] for c in msgs[-1]["content"] if c["type"] == "text"]
        assert "my chart caption" in final_texts


def _install_fake_deps(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = None
    fake_model.requires_grad_.return_value = None

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = MagicMock()
    fake_transformers.AutoProcessor.from_pretrained.return_value = MagicMock()
    fake_transformers.AutoModelForImageTextToText = MagicMock()
    fake_transformers.AutoModelForImageTextToText.from_pretrained.return_value = fake_model
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def _clear_qwen_module():
    for mod in list(sys.modules):
        if mod.startswith("agdg.targeting"):
            sys.modules.pop(mod, None)
