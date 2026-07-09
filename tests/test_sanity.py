import torch


def test_torch_version_and_device():
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    assert (major, minor) >= (2, 4)
    assert torch.backends.mps.is_available()
