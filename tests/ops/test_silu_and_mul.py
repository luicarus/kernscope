import math

import pytest
import torch

from kernscope import silu_and_mul


TOLERANCES = [(torch.float16, 1e-3), (torch.bfloat16, 1e-2), (torch.float32, 1e-6)]


def _reference(x: torch.Tensor) -> torch.Tensor:
    x64 = x.detach().cpu().double()
    hidden_size = x.shape[-1] // 2
    gate, up = x64.split(hidden_size, dim=-1)
    result = gate * torch.sigmoid(gate) * up
    return result.to(dtype=x.dtype, device=x.device)


@pytest.mark.parametrize("dtype,tol", TOLERANCES)
@pytest.mark.parametrize("shape", [(6,), (2, 3, 10), (0, 10)])
def test_torch_matches_fp64_reference(dtype, tol, shape):
    size = math.prod(shape)
    x = torch.linspace(-1.25, 2.0, size).reshape(shape).to(dtype)
    before = x.clone()

    actual = silu_and_mul(x)

    assert actual is not x
    assert actual.shape == (*x.shape[:-1], x.shape[-1] // 2)
    assert actual.dtype == x.dtype and actual.device == x.device
    torch.testing.assert_close(actual, _reference(x), rtol=tol, atol=tol)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


def test_torch_supports_autograd():
    x = torch.tensor(
        [[-2.0, -0.5, 0.5, 2.0, 1.5, -1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    silu_and_mul(x).sum().backward()

    gate, up = x.detach().double().split(3, dim=-1)
    sigmoid = torch.sigmoid(gate)
    expected = torch.cat(
        (up * sigmoid * (1 + gate * (1 - sigmoid)), gate * sigmoid), dim=-1
    ).to(x.dtype)
    assert x.grad is not None
    torch.testing.assert_close(x.grad, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype,tol", TOLERANCES)
def test_torch_handles_extreme_gate_values(dtype, tol):
    gate = torch.tensor([-80, -20, -5, 0, 5, 20, 80], dtype=dtype)
    x = torch.cat((gate, torch.full_like(gate, 0.5)))

    actual = silu_and_mul(x)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, _reference(x), rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "x,error",
    [
        ([1, 2, 3, 4], TypeError),
        (torch.tensor(1.0), ValueError),
        (torch.empty(2, 0), ValueError),
        (torch.ones(3), ValueError),
        (torch.ones(4, dtype=torch.int32), TypeError),
        (torch.ones(2, 4)[:, ::2], ValueError),
    ],
    ids=["non-tensor", "scalar", "empty-final-dim", "odd-final-dim", "dtype", "noncontiguous"],
)
def test_rejects_invalid_inputs(x, error):
    with pytest.raises(error):
        silu_and_mul(x)


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be 'torch' or 'triton'"):
        silu_and_mul(torch.ones(4), backend="unknown")


def test_triton_backend_requires_cuda():
    with pytest.raises(ValueError, match="requires CUDA"):
        silu_and_mul(torch.ones(4), backend="triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype,tol", TOLERANCES)
@pytest.mark.parametrize("shape", [(6,), (2, 3, 10), (2, 1026), (2, 4096), (0, 10)])
def test_triton_matches_torch(dtype, tol, shape):
    x = torch.linspace(-1.25, 2.0, math.prod(shape), device="cuda").reshape(shape).to(dtype)
    before = x.clone()

    actual = silu_and_mul(x, backend="triton")

    assert actual is not x
    assert actual.shape == (*x.shape[:-1], x.shape[-1] // 2)
    assert actual.dtype == x.dtype and actual.device == x.device
    torch.testing.assert_close(actual, silu_and_mul(x), rtol=tol, atol=tol)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    if x.numel():
        assert actual.data_ptr() != x.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype,tol", TOLERANCES)
def test_triton_accepts_contiguous_offset_view(dtype, tol):
    base = torch.linspace(-1.25, 2.0, 1 + 2 * 10, device="cuda").to(dtype)
    x = base[1:].view(2, 10)
    before = x.clone()

    assert x.is_contiguous() and x.data_ptr() != base.data_ptr()
    actual = silu_and_mul(x, backend="triton")

    assert actual.shape == (2, 5)
    torch.testing.assert_close(actual, _reference(x), rtol=tol, atol=tol)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype,tol", TOLERANCES)
def test_triton_handles_extreme_gate_values(dtype, tol):
    gate = torch.tensor([-80, -20, -5, 0, 5, 20, 80], device="cuda", dtype=dtype)
    x = torch.cat((gate, torch.full_like(gate, 0.5)))

    actual = silu_and_mul(x, backend="triton")

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, _reference(x), rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_backend_rejects_autograd():
    x = torch.ones(4, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="does not support autograd"):
        silu_and_mul(x, backend="triton")
