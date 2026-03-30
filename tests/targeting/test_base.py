"""Tests for the TargetingModel abstract base class."""
import pytest
from PIL import Image

from agdg.targeting.strategies.base import TargetingModel


def test_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        TargetingModel()


def test_incomplete_subclass_missing_call():
    class NoCall(TargetingModel):
        def __init__(self, model_tag=None):
            pass

    with pytest.raises(TypeError):
        NoCall()


def test_incomplete_subclass_missing_init():
    class NoInit(TargetingModel):
        def __call__(self, images, clean_texts):
            return []

    with pytest.raises(TypeError):
        NoInit()


def test_valid_subclass():
    class Dummy(TargetingModel):
        def __init__(self, model_tag=None):
            self.tag = model_tag

        def __call__(self, images, clean_texts):
            return [f"target for {t}" for t in clean_texts]

    d = Dummy(model_tag="test")
    assert d.tag == "test"

    img = Image.new("RGB", (10, 10))
    result = d([img], ["clean caption"])
    assert result == ["target for clean caption"]
