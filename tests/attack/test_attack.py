import types

import pytest
from PIL import Image

from agdg.attack import attack as attack_module

class FakeImage:
    def __init__(self):
        self.saved_path = None

    def save(self, path):
        self.saved_path = path


class FakeTargetModel:
    def __init__(self):
        self.created = True


class FakeTextAttacker:
    last_instance = None

    def __init__(self, model):
        self.model = model
        self.kwargs = None
        FakeTextAttacker.last_instance = self

    def attack(self, **kwargs):
        print("text")
        self.kwargs = kwargs
        clean = self.kwargs["clean"]
        if isinstance(clean, Image.Image):
            return FakeImage()
        else:
            return [FakeImage()] * len(clean)


class FakeImageAttacker:
    last_instance = None

    def __init__(self, model):
        self.model = model
        self.kwargs = None
        FakeImageAttacker.last_instance = self

    def attack(self, **kwargs):
        print("text")
        self.kwargs = kwargs
        self.kwargs = kwargs
        clean = self.kwargs["clean"]
        if isinstance(clean, Image.Image):
            return FakeImage()
        else:
            return [FakeImage()] * len(clean)


class FakeUntargetedAttacker:
    last_instance = None

    def __init__(self, model):
        self.model = model
        self.kwargs = None
        FakeUntargetedAttacker.last_instance = self

    def attack(self, **kwargs):
        print("text")
        self.kwargs = kwargs
        self.kwargs = kwargs
        clean = self.kwargs["clean"]
        if isinstance(clean, Image.Image):
            return FakeImage()
        else:
            return [FakeImage()] * len(clean)


def setup_fakes(monkeypatch):
    monkeypatch.setattr(attack_module, "TextCLIPModel", FakeTargetModel)
    monkeypatch.setattr(attack_module, "PatchTextCLIPModel", FakeTargetModel)
    monkeypatch.setattr(attack_module, "ImageCLIPModel", FakeTargetModel)
    monkeypatch.setattr(attack_module, "LlavaTextTargetModel", FakeTargetModel)
    monkeypatch.setattr(attack_module, "AttackVLMText", FakeTextAttacker)
    monkeypatch.setattr(attack_module, "AttackVLMOCR", FakeTextAttacker)
    monkeypatch.setattr(attack_module, "AttackVLMImage", FakeImageAttacker)
    monkeypatch.setattr(attack_module, "AttackVLMUntargeted", FakeUntargetedAttacker)
    monkeypatch.setattr(attack_module, "_get_device", lambda device=None: "cpu")
    monkeypatch.setattr(attack_module.Image, "open", lambda path: f"opened:{path}")


def test_attack_builds_text_target_and_source(monkeypatch):
    setup_fakes(monkeypatch)

    attack_module.attack(
        attacker="targeted_text",
        model="clip_text",
        clean_image_path="clean.png",
        target_question="Who won?",
        target_response="Ferrari",
        source_response="Mercedes",
    )

    kwargs = FakeTextAttacker.last_instance.kwargs
    assert kwargs["clean"] == "opened:clean.png"
    assert kwargs["target"] == "Question: Who won?\nAnswer: Ferrari"
    assert kwargs["strength"] == 1.0
    assert kwargs["hyperparameters"]["source_text"] == "Question: Who won?\nAnswer: Mercedes"


def test_attack_uses_target_image_for_image_mode(monkeypatch):
    setup_fakes(monkeypatch)

    attack_module.attack(
        attacker="targeted_image",
        model="clip_image",
        clean_image_path="clean.png",
    )

    kwargs = FakeImageAttacker.last_instance.kwargs
    assert kwargs["clean"] == "opened:clean.png"
    assert kwargs["target"] == "opened:/root/target.jpg"
    assert kwargs["strength"] == 1.0


def test_attack_uses_plain_response_when_question_missing(monkeypatch):
    setup_fakes(monkeypatch)

    attack_module.attack(
        attacker="targeted_text_ocr",
        model="clip_text_patch",
        clean_image_path="clean.png",
        target_question="",
        target_response="16",
        source_response="15",
    )

    kwargs = FakeTextAttacker.last_instance.kwargs
    assert kwargs["target"] == "16"
    assert kwargs["hyperparameters"]["source_text"] == "15"


def test_attack_leaves_source_text_empty_when_not_provided(monkeypatch):
    setup_fakes(monkeypatch)

    attack_module.attack(
        attacker="targeted_text",
        model="clip_text",
        clean_image_path="clean.png",
        target_question="Who won?",
        target_response="Ferrari",
        source_response="",
    )

    kwargs = FakeTextAttacker.last_instance.kwargs
    assert kwargs["target"] == "Question: Who won?\nAnswer: Ferrari"
    assert "source_text" not in kwargs["hyperparameters"]


def test_get_device_returns_explicit_device():
    assert attack_module._get_device("my-device") == "my-device"


def test_get_device_auto_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert attack_module._get_device() == "cpu"


def test_get_device_auto_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert attack_module._get_device() == "cuda"


def test_get_device_auto_mps(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert attack_module._get_device() == "mps"


def test_ensure_dependencies_loaded(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(attack_module, "AttackVLMImage", None)
    monkeypatch.setattr(attack_module, "AttackVLMOCR", None)
    monkeypatch.setattr(attack_module, "AttackVLMText", None)
    monkeypatch.setattr(attack_module, "AttackVLMUntargeted", None)
    monkeypatch.setattr(attack_module, "ImageCLIPModel", None)
    monkeypatch.setattr(attack_module, "PatchTextCLIPModel", None)
    monkeypatch.setattr(attack_module, "TextCLIPModel", None)
    monkeypatch.setattr(attack_module, "LlavaTextTargetModel", None)

    attack_module._ensure_dependencies_loaded()

    assert attack_module.AttackVLMImage is not None
    assert attack_module.TextCLIPModel is not None
    assert attack_module.LlavaTextTargetModel is not None


def test_build_target_model_unknown(monkeypatch):
    setup_fakes(monkeypatch)
    with pytest.raises(ValueError, match="Unknown surrogate model"):
        attack_module.build_target_model("nonexistent_model")


def test_build_attack_method_unknown(monkeypatch):
    setup_fakes(monkeypatch)
    with pytest.raises(ValueError, match="Unknown attacker"):
        attack_module.build_attack_method("nonexistent_attacker", "clip_text")


def test_generate_adversarial_untargeted(monkeypatch):
    setup_fakes(monkeypatch)
    result = attack_module.generate_adversarial_image(
        attacker="untargeted",
        model="clip_image",
        clean_image_path="clean.png",
    )
    assert isinstance(result, list)
    kwargs = FakeUntargetedAttacker.last_instance.kwargs
    assert kwargs["clean"] == "opened:clean.png"
    assert kwargs["strength"] == 1.0


def test_attack_saves_single_image(monkeypatch):
    setup_fakes(monkeypatch)

    saved_paths = []

    class PilImageAttacker:
        def __init__(self, model):
            self.model = model

        def attack(self, **kwargs):
            return Image.new("RGB", (10, 10))

    monkeypatch.setattr(attack_module, "AttackVLMText", PilImageAttacker)
    monkeypatch.setattr(Image.Image, "save", lambda self, path, **kw: saved_paths.append(path))

    attack_module.attack(
        attacker="targeted_text",
        model="clip_text",
        clean_image_path="clean.png",
    )

    assert saved_paths == ["/root/images/adversarial_output.png"]
