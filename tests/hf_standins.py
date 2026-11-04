"""Stand-in `torch` and `transformers` modules for verifier and training tests.

Nothing here imports the real packages (conftest.py would fail the test if anything
did). The stand-ins are just rich enough for the code under test: numpy-backed tensors,
a tokenizer that remembers its inputs, and classifiers whose logits come from a simple
lexical rule, so tests can predict the outcome.
"""

import sys
import types
from pathlib import Path

import numpy as np

from groundcheck.verify.spans import numbers, overlap_score


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float64)

    def __getitem__(self, index):
        return FakeTensor(self.data[index])

    @property
    def shape(self):
        return self.data.shape

    def float(self):
        return self

    def cpu(self):
        return self

    def to(self, device):
        return self

    def tolist(self):
        return self.data.tolist()


class FakeBatch(dict):
    def to(self, device):
        self["device"] = device
        return self


class NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def softmax(tensor, dim=-1):
    x = tensor.data - tensor.data.max(axis=dim, keepdims=True)
    e = np.exp(x)
    return FakeTensor(e / e.sum(axis=dim, keepdims=True))


def make_torch(cuda=False, mps=False):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.5.0-standin"
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    torch.no_grad = NoGrad
    torch.softmax = softmax
    return torch


def support_logit(claim, text):
    """> 0 when the text covers the claim's words and numbers (same rule as FakeVerifier)."""
    covered = overlap_score(claim, text) >= 0.8 and numbers(claim) <= numbers(text)
    return 3.0 if covered else -3.0


def contradiction_logit(claim, text):
    claim_numbers = numbers(claim)
    near = overlap_score(claim, text) >= 0.4
    return 3.0 if claim_numbers and near and not claim_numbers <= numbers(text) else -3.0


class Recorder:
    def __init__(self):
        self.loads = []  # (class name, model name, kwargs)
        self.tokenized = []  # (texts, claims, kwargs)
        self.devices = []
        self.predicted = []  # HHEM pairs


def make_transformers(recorder, id2label=None, three_way=False):
    transformers = types.ModuleType("transformers")
    transformers.__version__ = "4.50.0-standin"

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            recorder.loads.append(("AutoTokenizer", name, kwargs))
            return cls()

        def __call__(self, texts, claims=None, **kwargs):
            recorder.tokenized.append((list(texts), list(claims or []), kwargs))
            return FakeBatch(texts=list(texts), claims=list(claims or []))

    class Classifier:
        def __init__(self, name):
            self.name = name
            if three_way:
                labels = id2label or {0: "supported", 1: "contradicted", 2: "unsupported"}
            else:
                labels = id2label or {0: "LABEL_0", 1: "LABEL_1"}
            self.config = types.SimpleNamespace(id2label=labels, _commit_hash="commit123")

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            recorder.loads.append(("AutoModelForSequenceClassification", name, kwargs))
            return cls(name)

        def to(self, device):
            recorder.devices.append(device)
            return self

        def eval(self):
            return self

        def __call__(self, texts, claims, **kwargs):
            rows = []
            for text, claim in zip(texts, claims, strict=True):
                s, c = support_logit(claim, text), contradiction_logit(claim, text)
                if three_way:
                    labels = self.config.id2label
                    by_name = {"supported": s, "contradicted": c, "unsupported": 0.0}
                    rows.append([by_name[labels[i]] for i in range(3)])
                else:
                    rows.append([-s, s])
            return types.SimpleNamespace(logits=FakeTensor(rows))

        def predict(self, pairs):  # HHEM's custom method: consistency score per pair
            recorder.predicted.append(list(pairs))
            return FakeTensor([0.9 if support_logit(h, p) > 0 else 0.1 for p, h in pairs])

    transformers.AutoTokenizer = Tokenizer
    transformers.AutoModelForSequenceClassification = Classifier
    return transformers


def install(monkeypatch, cuda=False, mps=False, id2label=None, three_way=False):
    """Put the stand-ins in sys.modules for one test; returns the recorder."""
    recorder = Recorder()
    monkeypatch.setitem(sys.modules, "torch", make_torch(cuda=cuda, mps=mps))
    monkeypatch.setitem(
        sys.modules, "transformers", make_transformers(recorder, id2label, three_way)
    )
    return recorder


def install_training(monkeypatch, cuda=False):
    """Stand-ins for the parts of transformers that training.train uses. The fake
    Trainer 'learns' perfectly: evaluate() returns one-hot logits of the true labels."""
    import json as _json

    recorder = Recorder()
    recorder.seeds, recorder.args, recorder.trainer = [], {}, {}
    transformers = types.ModuleType("transformers")
    transformers.__version__ = "4.50.0-standin"

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            recorder.loads.append(("AutoTokenizer", name, kwargs))
            return cls()

        def __call__(self, first, second=None, **kwargs):
            recorder.tokenized.append((first, second, kwargs))
            ids = list(range(min(len(first.split()) + len((second or "").split()), 8)))
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        def save_pretrained(self, path):
            (Path(path) / "tokenizer_config.json").write_text("{}")

    class Model:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            recorder.loads.append(("AutoModelForSequenceClassification", name, kwargs))
            return cls(kwargs)

    class TrainingArguments:
        def __init__(self, **kwargs):
            recorder.args.update(kwargs)
            self.__dict__.update(kwargs)

    class Trainer:
        def __init__(self, **kwargs):
            recorder.trainer.update(kwargs)
            self.model, self.compute_metrics = kwargs["model"], kwargs["compute_metrics"]

        def train(self):
            return types.SimpleNamespace(training_loss=0.42)

        def evaluate(self, dataset, metric_key_prefix="eval"):
            labels = np.array([row["labels"] for row in dataset.rows])
            logits = np.eye(3)[labels]
            metrics = self.compute_metrics(
                types.SimpleNamespace(predictions=logits, label_ids=labels)
            )
            out = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
            out[f"{metric_key_prefix}_loss"] = 0.1
            return out

        def save_model(self, path):
            config = {"id2label": {str(k): v for k, v in self.model.kwargs["id2label"].items()}}
            (Path(path) / "config.json").write_text(_json.dumps(config))

    transformers.AutoTokenizer = Tokenizer
    transformers.AutoModelForSequenceClassification = Model
    transformers.TrainingArguments = TrainingArguments
    transformers.Trainer = Trainer
    transformers.DataCollatorWithPadding = lambda tokenizer: ("collator", tokenizer)
    transformers.set_seed = recorder.seeds.append
    monkeypatch.setitem(sys.modules, "torch", make_torch(cuda=cuda))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return recorder
