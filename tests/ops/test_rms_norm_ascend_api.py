import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from kernscope import rms_norm


@pytest.fixture
def npu_inputs():
    """Mock device metadata for API checks without allocating NPU tensors."""
    properties = dict(shape=(4,), ndim=1, dtype=torch.float32,
                      device=SimpleNamespace(type="npu"), requires_grad=False)
    inputs = tuple(Mock(spec=torch.Tensor, **properties) for _ in range(2))
    for tensor in inputs:
        tensor.is_contiguous.return_value = True
        tensor.numel.return_value = 4
    return inputs


def test_dispatches_to_ascend_backend(monkeypatch, npu_inputs):
    module = ModuleType("kernscope.backends.tilelang_ascend")
    expected = torch.empty(4)
    module.rms_norm_tilelang_ascend = Mock(return_value=expected)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    actual = rms_norm(*npu_inputs, eps=1e-5, backend="tilelang_ascend")

    assert actual is expected
    module.rms_norm_tilelang_ascend.assert_called_once_with(*npu_inputs, 1e-5)


def test_ascend_reports_missing_dependencies(monkeypatch, npu_inputs):
    monkeypatch.delitem(sys.modules, "kernscope.backends.tilelang_ascend", raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    with pytest.raises(ModuleNotFoundError) as error:
        rms_norm(*npu_inputs, backend="tilelang_ascend")
    assert error.value.name == "torch_npu"


@pytest.mark.parametrize("input_index", [0, 1], ids=["x", "weight"])
def test_ascend_rejects_autograd(npu_inputs, input_index):
    npu_inputs[input_index].requires_grad = True
    with pytest.raises(ValueError, match="tilelang_ascend backend does not support autograd"):
        rms_norm(*npu_inputs, backend="tilelang_ascend")


def test_torch_runs_without_ascend_dependencies(monkeypatch):
    for module in ("tilelang", "torch_npu", "kernscope.backends.tilelang_ascend"):
        monkeypatch.setitem(sys.modules, module, None)
    x = torch.ones(4)
    torch.testing.assert_close(rms_norm(x, x), torch.full_like(x, (1 + 1e-6) ** -0.5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ascend_rejects_cuda():
    x = torch.ones(4, device="cuda")
    with pytest.raises(ValueError, match="requires NPU"):
        rms_norm(x, x, backend="tilelang_ascend")
