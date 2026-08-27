import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import SimpleCNN, get_model, get_resnet18  # noqa: E402


def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_resnet18_output_shape():
    model = get_resnet18(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_get_model_resnet18():
    model = get_model("resnet18", num_classes=10)
    assert model.fc.out_features == 10


def test_get_model_cnn():
    model = get_model("cnn", num_classes=10)
    assert isinstance(model, SimpleCNN)


def test_get_model_unknown_architecture_raises():
    try:
        get_model("not-a-real-architecture", num_classes=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_model_is_trainable_single_step():
    model = SimpleCNN(num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
