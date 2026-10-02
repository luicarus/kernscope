import math

import pytest
import torch

from kernscope import rms_norm


@pytest.mark.parametrize(
    "dtype,tol", [(torch.float16, 1e-3), (torch.bfloat16, 1e-2), (torch.float32, 1e-6)]
)
@pytest.mark.parametrize("shape", [(5,), (2, 3, 5)])
def test_matches_fp64_reference(dtype, tol, shape):
    eps = 1e-5
    x = torch.linspace(-1.25, 2.0, math.prod(shape)).reshape(shape).to(dtype)
    weight = torch.linspace(0.7, 1.3, shape[-1]).to(dtype)
    x64, weight64 = x.double(), weight.double()
    scale = torch.rsqrt(x64.square().mean(-1, keepdim=True) + eps)
    expected = (x64 * scale * weight64).to(dtype)

    torch.testing.assert_close(rms_norm(x, weight, eps), expected, rtol=tol, atol=tol)


def test_matches_hand_calculated_example():
    actual = rms_norm(torch.tensor([3.0, 4.0]), torch.ones(2))
    expected = torch.tensor([0.8485281, 1.1313709])

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "x,weight,error",
    [
        (torch.tensor(1.0), torch.ones(1), ValueError),
        (torch.ones(4), torch.ones(3), ValueError),
        (torch.ones(4, dtype=torch.int32), torch.ones(4, dtype=torch.int32), TypeError),
        (torch.ones(4), torch.ones(4, dtype=torch.float16), ValueError),
        (torch.ones(2, 4)[:, ::2], torch.ones(2), ValueError),
        (torch.ones(2, 2), torch.ones(4)[::2], ValueError),
    ],
    ids=["scalar", "weight-shape", "dtype", "weight-dtype", "strided-x", "strided-weight"],
)
@pytest.mark.parametrize("backend", ["torch", "triton", "tilelang_ascend"])
def test_rejects_invalid_inputs(x, weight, error, backend):
    with pytest.raises(error):
        rms_norm(x, weight, backend=backend)


@pytest.mark.parametrize(
    "eps,error",
    [
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("1e-6", TypeError),
    ],
)
@pytest.mark.parametrize("backend", ["torch", "triton", "tilelang_ascend"])
def test_rejects_invalid_epsilon(eps, error, backend):
    with pytest.raises(error):
        rms_norm(torch.ones(4), torch.ones(4), eps, backend=backend)


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be 'torch', 'triton', or 'tilelang_ascend'"):
        rms_norm(torch.ones(4), torch.ones(4), backend="unknown")


@pytest.mark.parametrize("backend,device", [("triton", "CUDA"), ("tilelang_ascend", "NPU")])
def test_accelerator_backend_requires_device(backend, device):
    with pytest.raises(ValueError, match=f"requires {device}"):
        rms_norm(torch.ones(4), torch.ones(4), backend=backend)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "dtype,tol",
    [(torch.float16, 1e-3), (torch.bfloat16, 1e-2), (torch.float32, 1e-6)],
)
@pytest.mark.parametrize("shape", [(5,), (2, 3, 5), (2, 1024), (3, 1000), (512, 1024)])
def test_triton_matches_torch(dtype, tol, shape):
    eps = 1e-5
    x = torch.linspace(-1.25, 2.0, math.prod(shape)).reshape(shape)
    x = x.to(device="cuda", dtype=dtype)
    weight = torch.linspace(0.7, 1.3, shape[-1]).to(device="cuda", dtype=dtype)

    expected = rms_norm(x, weight, eps)
    actual = rms_norm(x, weight, eps, backend="triton")

    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_backend_rejects_autograd():
    x = torch.ones(4, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="does not support autograd"):
        rms_norm(x, torch.ones(4, device="cuda"), backend="triton")
