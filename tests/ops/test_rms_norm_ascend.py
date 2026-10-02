import math

import pytest
import torch

from kernscope import rms_norm


@pytest.fixture(scope="module")
def npu_device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is required")
    pytest.importorskip("tilelang")
    return torch.device("npu")


@pytest.mark.parametrize(
    "dtype,shape,tol",
    [
        (torch.float16, (5,), 1e-3),
        (torch.bfloat16, (2, 3, 5), 1e-2),
        (torch.float32, (3, 1000), 1e-6),
        (torch.bfloat16, (512, 1024), 1e-2),
    ],
)
def test_ascend_matches_reference(npu_device, dtype, shape, tol):
    x_cpu = torch.linspace(-1.25, 2.0, math.prod(shape)).reshape(shape).to(dtype)
    weight_cpu = torch.linspace(0.7, 1.3, shape[-1]).to(dtype)
    x, weight = x_cpu.to(npu_device), weight_cpu.to(npu_device)

    actual = rms_norm(x, weight, eps=1e-5, backend="tilelang_ascend")

    assert actual.shape == x.shape and actual.dtype == x.dtype and actual.device == x.device
    assert actual.data_ptr() != x.data_ptr()
    torch.testing.assert_close(actual.cpu(), rms_norm(x_cpu, weight_cpu, 1e-5), rtol=tol, atol=tol)
    torch.testing.assert_close(x.cpu(), x_cpu, rtol=0, atol=0)
    torch.testing.assert_close(weight.cpu(), weight_cpu, rtol=0, atol=0)


def test_ascend_zero_input(npu_device):
    x = torch.zeros((512, 1024), device=npu_device, dtype=torch.bfloat16)
    weight = torch.ones(1024, device=npu_device, dtype=x.dtype)

    actual = rms_norm(x, weight, eps=1e-5, backend="tilelang_ascend")

    torch.testing.assert_close(actual.cpu(), torch.zeros_like(x, device="cpu"), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0, 5), (2, 0, 5)])
def test_ascend_empty_input(npu_device, shape):
    x = torch.empty(shape, device=npu_device, dtype=torch.bfloat16)
    weight = torch.ones(shape[-1], device=npu_device, dtype=x.dtype)

    actual = rms_norm(x, weight, backend="tilelang_ascend")

    assert actual.shape == x.shape and actual.dtype == x.dtype and actual.device == x.device
    assert actual.numel() == 0
