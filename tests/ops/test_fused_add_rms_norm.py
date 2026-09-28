import math

import pytest
import torch

from kernscope import fused_add_rms_norm

TOLERANCES = [(torch.float16, 1e-3), (torch.bfloat16, 1e-2), (torch.float32, 1e-6)]


@pytest.mark.parametrize("dtype,tol", TOLERANCES)
@pytest.mark.parametrize("shape", [(5,), (2, 3, 5)])
def test_updates_residual_and_normalizes_sum_in_place(dtype, tol, shape):
    eps = 1e-5
    x = torch.linspace(-1.25, 2.0, math.prod(shape)).reshape(shape).to(dtype)
    residual = torch.linspace(0.5, -0.5, math.prod(shape)).reshape(shape).to(dtype)
    weight = torch.linspace(0.7, 1.3, shape[-1]).to(dtype)
    x64, residual64, weight64 = x.double(), residual.double(), weight.double()
    summed = x64 + residual64
    expected_residual = summed.to(dtype)
    scale = torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + eps)
    expected_x = (summed * scale * weight64).to(dtype)
    x_ptr, residual_ptr = x.data_ptr(), residual.data_ptr()

    result = fused_add_rms_norm(x, residual, weight, eps)

    assert result is None
    assert x.data_ptr() == x_ptr and residual.data_ptr() == residual_ptr
    torch.testing.assert_close(x, expected_x, rtol=tol, atol=tol)
    torch.testing.assert_close(residual, expected_residual, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "x,residual,weight,error",
    [
        (torch.ones(4), torch.ones(3), torch.ones(4), ValueError),
        (torch.ones(4), torch.ones(4), torch.ones(3), ValueError),
        (torch.ones(4), torch.ones(4, dtype=torch.float16), torch.ones(4), ValueError),
        (
            torch.ones(4, dtype=torch.int32),
            torch.ones(4, dtype=torch.int32),
            torch.ones(4, dtype=torch.int32),
            TypeError,
        ),
        (torch.ones(2, 2), torch.ones(2, 4)[:, ::2], torch.ones(2), ValueError),
    ],
    ids=["residual-shape", "weight-shape", "residual-dtype", "dtype", "noncontiguous-residual"],
)
def test_rejects_invalid_inputs(x, residual, weight, error):
    with pytest.raises(error):
        fused_add_rms_norm(x, residual, weight)


def test_triton_backend_requires_cuda():
    with pytest.raises(ValueError, match="requires CUDA"):
        fused_add_rms_norm(torch.ones(4), torch.ones(4), torch.ones(4), backend="triton")


@pytest.mark.parametrize(
    "eps,error", [(0.0, ValueError), (float("nan"), ValueError), (True, TypeError)]
)
def test_rejects_invalid_epsilon(eps, error):
    with pytest.raises(error):
        fused_add_rms_norm(torch.ones(4), torch.ones(4), torch.ones(4), eps)


def test_rejects_overlapping_x_and_residual():
    storage = torch.arange(6, dtype=torch.float32)
    with pytest.raises(ValueError, match="must not overlap"):
        fused_add_rms_norm(storage[:4], storage[2:], torch.ones(4))


def test_rejects_overlapping_weight():
    storage = torch.arange(8, dtype=torch.float32)
    with pytest.raises(ValueError, match="must not overlap"):
        fused_add_rms_norm(storage[:4], torch.ones(4), storage[2:6])


def test_rejects_autograd():
    with pytest.raises(ValueError, match="inference-only"):
        fused_add_rms_norm(torch.ones(4, requires_grad=True), torch.ones(4), torch.ones(4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype,tol", TOLERANCES)
@pytest.mark.parametrize("shape", [(5,), (2, 3, 5), (2, 1024), (3, 1000)])
def test_triton_matches_torch(dtype, tol, shape):
    eps = 1e-5
    x = torch.linspace(-1.25, 2.0, math.prod(shape)).reshape(shape)
    residual = torch.linspace(0.5, -0.5, math.prod(shape)).reshape(shape)
    x = x.to(device="cuda", dtype=dtype)
    residual = residual.to(device="cuda", dtype=dtype)
    weight = torch.linspace(0.7, 1.3, shape[-1]).to(device="cuda", dtype=dtype)
    expected_x, expected_residual = x.clone(), residual.clone()
    actual_x, actual_residual = x.clone(), residual.clone()

    fused_add_rms_norm(expected_x, expected_residual, weight, eps)
    result = fused_add_rms_norm(actual_x, actual_residual, weight, eps, backend="triton")

    assert result is None
    torch.testing.assert_close(actual_x, expected_x, rtol=tol, atol=tol)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=tol, atol=tol)
