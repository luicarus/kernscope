import math
import sys

import pytest
import torch

from kernscope import softmax

TOLERANCES = [(torch.float16, 1e-3), (torch.bfloat16, 1e-2), (torch.float32, 1e-6)]
BACKENDS = [("torch", "cpu"), pytest.param("triton", "cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"))]


def _reference(x):
    return torch.softmax(x.detach().cpu().double(), -1).to(dtype=x.dtype, device=x.device)


@pytest.mark.parametrize("dtype,tol", TOLERANCES)
@pytest.mark.parametrize("shape", [(1,), (7,), (2, 3, 5), (0, 7), (2, 0, 5), (4, 128),
                                  (3, 129), (4, 256), (3, 257), (3, 1000),
                                  (4, 1024), (2, 4096), (3, 4097), (2, 8192), (1, 16384),
                                  (513, 127), (513, 129), (512, 256), (513, 257),
                                  (511, 1024), (513, 1023), (512, 1024), (513, 1024), (512, 1025)])
@pytest.mark.parametrize("backend,device", BACKENDS)
def test_matches_fp64_reference(dtype, tol, shape, backend, device):
    x = torch.linspace(-1.25, 2.0, math.prod(shape), device=device).reshape(shape).to(dtype)
    before = x.clone()
    expected = _reference(x)

    actual = softmax(x, backend=backend)

    assert actual is not x
    assert actual.shape == x.shape and actual.dtype == x.dtype and actual.device == x.device
    if x.numel():
        assert actual.data_ptr() != x.data_ptr()
        torch.testing.assert_close(actual.float().sum(-1), torch.ones(shape[:-1], device=device), rtol=tol, atol=tol)
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    if backend == "triton":
        torch.testing.assert_close(actual, softmax(x), rtol=tol, atol=tol)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


def test_matches_hand_calculated_example():
    x = torch.tensor([0.0, math.log(2), -math.inf])
    torch.testing.assert_close(softmax(x), torch.tensor([1 / 3, 2 / 3, 0.0]), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype,tol", TOLERANCES)
@pytest.mark.parametrize("backend,device", BACKENDS)
def test_large_logits_remain_finite(dtype, tol, backend, device):
    x = torch.tensor([[10000, 10001, -10000], [-10000, -10001, -9999]], dtype=dtype, device=device)

    actual = softmax(x, backend=backend)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, _reference(x), rtol=tol, atol=tol)


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_uses_fp32_denominator(backend, device):
    x = torch.zeros(65536, dtype=torch.float16, device=device)
    torch.testing.assert_close(softmax(x, backend=backend), torch.full_like(x, 1 / x.numel()), rtol=0, atol=0)


@pytest.mark.parametrize("dtype,tol", TOLERANCES)
@pytest.mark.parametrize("backend,device", BACKENDS)
@pytest.mark.parametrize("repeats", [1, 129])
def test_nonfinite_rows_follow_torch(dtype, tol, backend, device, repeats):
    x = torch.tensor([[0, -math.inf, 1], [math.inf, 0, -1],
                      [-math.inf, -math.inf, -math.inf], [math.nan, 1, 2]], dtype=dtype, device=device).repeat(repeats, 1)
    before = x.clone()

    actual = softmax(x, backend=backend)

    grouped = actual.view(repeats, 4, 3)
    assert (grouped[:, 0, 1] == 0).all()
    assert torch.isnan(grouped[:, 1:]).all()
    torch.testing.assert_close(actual, _reference(x), rtol=tol, atol=tol, equal_nan=True)
    torch.testing.assert_close(x, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype,tol", TOLERANCES)
def test_supports_autograd(dtype, tol):
    x = torch.tensor([[0.5, -1, 2], [-2, 0, 1]], dtype=dtype, requires_grad=True)
    grad = torch.tensor([[1, 2, -1], [0.5, -0.25, 3]], dtype=dtype)

    softmax(x).backward(grad)

    p = torch.softmax(x.detach().double(), -1)
    g = grad.double()
    expected = (p * (g - (p * g).sum(-1, keepdim=True))).to(dtype)
    torch.testing.assert_close(x.grad, expected, rtol=tol, atol=tol)


def test_empty_batch_supports_autograd():
    x = torch.empty(0, 5, requires_grad=True)
    softmax(x).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


@pytest.mark.parametrize("device,amp_dtype", [("cpu", torch.bfloat16), pytest.param(
    "cuda", torch.float16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"))])
def test_preserves_fp32_under_autocast(device, amp_dtype):
    x = torch.tensor([0.5, -1.0, 2.0], device=device)
    with torch.autocast(device, dtype=amp_dtype):
        actual = softmax(x)
        if device == "cuda":
            torch.testing.assert_close(softmax(x, backend="triton"), actual, rtol=1e-6, atol=1e-6)
        assert torch.is_autocast_enabled(device)
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, _reference(x), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype,tol", TOLERANCES)
@pytest.mark.parametrize("backend,device", BACKENDS)
@pytest.mark.parametrize("shape", [(2, 5), (513, 129)])
def test_accepts_contiguous_offset_view(dtype, tol, backend, device, shape):
    base = torch.linspace(-1.25, 2.0, math.prod(shape) + 1, device=device).to(dtype)
    x = base[1:].view(shape)
    before = base.clone()

    actual = softmax(x, backend=backend)

    assert x.is_contiguous() and x.storage_offset() == 1
    torch.testing.assert_close(actual, _reference(x), rtol=tol, atol=tol)
    torch.testing.assert_close(base, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "x,error,message",
    [
        ([1, 2], TypeError, "torch.Tensor"),
        (torch.tensor(1.0), ValueError, "nonempty final dimension"),
        (torch.empty(2, 0), ValueError, "nonempty final dimension"),
        (torch.ones(3, dtype=torch.int32), TypeError, "float16"),
        (torch.ones(3, dtype=torch.float64), TypeError, "float16"),
        (torch.ones(2, 6)[:, ::2], ValueError, "contiguous"),
    ],
    ids=["non-tensor", "scalar", "empty-final-dim", "integer", "float64", "strided"],
)
@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_rejects_invalid_inputs(x, error, message, backend):
    with pytest.raises(error, match=message):
        softmax(x, backend=backend)


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be 'torch' or 'triton'"):
        softmax(torch.ones(3), backend="unknown")


def test_triton_requires_cuda():
    with pytest.raises(ValueError, match="requires CUDA"):
        softmax(torch.ones(3), backend="triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_rejects_autograd():
    with pytest.raises(ValueError, match="does not support autograd"):
        softmax(torch.ones(2, 5, device="cuda", requires_grad=True), backend="triton")


def test_torch_runs_without_triton(monkeypatch):
    for module in ("triton", "kernscope.backends.triton_softmax"):
        monkeypatch.setitem(sys.modules, module, None)
    x = torch.ones(3)
    torch.testing.assert_close(softmax(x), torch.full_like(x, 1 / 3), rtol=1e-6, atol=1e-6)
