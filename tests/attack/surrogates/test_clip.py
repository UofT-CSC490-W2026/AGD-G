import types
import sys

import torch

from agdg.attack.surrogates.clip import CLIPModel, ImageCLIPModel, PatchTextCLIPModel, TextCLIPModel


def build_base_model(image_size=8):
    model = object.__new__(CLIPModel)
    model.clip_model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            vision_config=types.SimpleNamespace(image_size=image_size)
        )
    )
    return model


def test_get_image_size_uses_clip_config():
    model = build_base_model(image_size=12)

    assert model.get_image_size() == (12, 12)


def test_preprocess_resizes_images():
    model = build_base_model(image_size=8)
    images = torch.rand(1, 3, 4, 4)

    preprocessed = model._preprocess(images)

    assert preprocessed.shape == (1, 3, 8, 8)


def test_text_clip_model_returns_cosine_similarity_mean():
    model = object.__new__(TextCLIPModel)
    image = torch.tensor([[1.0, 0.0]])
    text = torch.tensor([[1.0, 0.0]])

    similarity = model(image, text)

    assert torch.isclose(similarity, torch.tensor(1.0))


def test_patch_text_clip_model_uses_best_patch():
    model = object.__new__(PatchTextCLIPModel)
    image = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    text = torch.tensor([[1.0, 0.0]])

    similarity = model(image, text)

    assert torch.isclose(similarity, torch.tensor(1.0))


def test_image_clip_model_averages_feature_similarities():
    model = object.__new__(ImageCLIPModel)
    image1 = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]
    image2 = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]

    similarity = model(image1, image2)

    assert torch.isclose(similarity, torch.tensor(1.0))


class FakeBatch(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeTokenizer:
    def __call__(self, texts, return_tensors=None, padding=None, truncation=None):
        return FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})


class FakeClipProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()


class FakeTextModel:
    def __call__(self, **kwargs):
        return types.SimpleNamespace(pooler_output=torch.tensor([[3.0, 4.0]]))


class FakeVisionModelHidden:
    def __call__(self, pixel_values=None, output_hidden_states=False, return_dict=True):
        hidden = [torch.tensor([[[0.0, 0.0], [3.0, 4.0], [0.0, 5.0]]], dtype=torch.float32)] * 24
        return types.SimpleNamespace(
            hidden_states=hidden,
            last_hidden_state=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [4.0, 3.0]]], dtype=torch.float32),
        )


class FakeVisionModelDense:
    def __call__(self, pixel_values=None, return_dict=True):
        return types.SimpleNamespace(
            last_hidden_state=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [4.0, 3.0]]], dtype=torch.float32)
        )


class FakeTransformersClipModel:
    def __init__(self):
        self.config = types.SimpleNamespace(vision_config=types.SimpleNamespace(image_size=8))
        self.vision_model = FakeVisionModelHidden()
        self.text_model = FakeTextModel()

    def to(self, device):
        return self

    def requires_grad_(self, flag):
        self.frozen = flag
        return self

    def visual_projection(self, tensor):
        return tensor

    def text_projection(self, tensor):
        return tensor


def install_fake_transformers(monkeypatch, *, clip_model=None, clip_processor=None):
    fake_transformers = types.ModuleType("transformers")

    class FakeClipModelClass:
        @staticmethod
        def from_pretrained(model_id, torch_dtype=None):
            return clip_model or FakeTransformersClipModel()

    class FakeClipProcessorClass:
        @staticmethod
        def from_pretrained(model_id):
            return clip_processor or FakeClipProcessor()

    fake_transformers.CLIPModel = FakeClipModelClass
    fake_transformers.CLIPProcessor = FakeClipProcessorClass
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_clip_model_init_loads_transformers_components(monkeypatch):
    install_fake_transformers(monkeypatch)

    model = CLIPModel(device="cpu", model_id="fake-model")

    assert model.device == "cpu"
    assert model.image_size == 8
    assert model.vision_model is model.clip_model.vision_model


def test_text_clip_model_embed_image_returns_normalized_projection(monkeypatch):
    install_fake_transformers(monkeypatch)
    model = TextCLIPModel(device="cpu", model_id="fake-model")
    image = torch.rand(1, 3, 4, 4)

    embedded = model.embed_image(image, detach=True)

    assert embedded.shape == (1, 2)
    assert torch.allclose(torch.norm(embedded, dim=-1), torch.ones(1))
    assert embedded.requires_grad is False


def test_text_clip_model_embed_text_returns_normalized_text_embedding(monkeypatch):
    install_fake_transformers(monkeypatch)
    model = TextCLIPModel(device="cpu", model_id="fake-model")

    embedded = model.embed_text("Ferrari", detach=True)

    assert embedded.shape == (1, 2)
    assert torch.allclose(torch.norm(embedded, dim=-1), torch.ones(1))
    assert embedded.requires_grad is False


def test_patch_text_clip_model_embed_image_returns_patch_embeddings(monkeypatch):
    clip_model = FakeTransformersClipModel()
    clip_model.vision_model = FakeVisionModelDense()
    install_fake_transformers(monkeypatch, clip_model=clip_model)
    model = PatchTextCLIPModel(device="cpu", model_id="fake-model")
    image = torch.rand(1, 3, 4, 4)

    embedded = model.embed_image(image, detach=True)

    assert embedded.shape == (1, 2, 2)
    norms = torch.norm(embedded, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms))


def test_image_clip_model_embed_image_returns_feature_list(monkeypatch):
    install_fake_transformers(monkeypatch)
    model = ImageCLIPModel(device="cpu", model_id="fake-model")
    image = torch.rand(1, 3, 4, 4)

    features = model.embed_image(image, detach=True)

    assert len(features) == 6
    assert all(feature.shape == (1, 2) for feature in features)
    assert all(torch.allclose(torch.norm(feature, dim=-1), torch.ones(1)) for feature in features)
