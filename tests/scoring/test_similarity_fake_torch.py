import math
import builtins
import sys
import types


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def unsqueeze(self, dim):
        return self


class FakeScore:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a.values, b.values))
    norm_a = math.sqrt(sum(x * x for x in a.values))
    norm_b = math.sqrt(sum(x * x for x in b.values))
    return FakeScore(dot / (norm_a * norm_b))


def test_get_device_with_fake_torch(monkeypatch):
    from agdg.scoring.similarity import get_device

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        device=lambda name: types.SimpleNamespace(type=name),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert get_device().type == "cpu"


def test_evaluate_similarity_with_fake_torch(monkeypatch):
    from agdg.scoring.similarity import evaluate_similarity

    fake_f = types.ModuleType("torch.nn.functional")
    fake_f.cosine_similarity = _cosine_similarity
    fake_nn = types.ModuleType("torch.nn")
    fake_nn.functional = fake_f
    fake_torch = types.ModuleType("torch")
    fake_torch.nn = fake_nn
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("torch"):
            if fromlist:
                if name == "torch.nn.functional":
                    return fake_f
                if name == "torch.nn":
                    return fake_nn
            return fake_torch
        if name == "torch.nn.functional":
            return fake_f
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    model = types.SimpleNamespace(
        encode=lambda texts, convert_to_tensor=True: [
            FakeTensor([1.0, 0.0]),
            FakeTensor([1.0, 0.0]),
            FakeTensor([0.0, 1.0]),
        ]
    )

    winner = evaluate_similarity(model, "output", "match-a", "match-b")

    assert winner == "A"
