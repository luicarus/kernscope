import math

import pytest
import torch

from kernscope import gemv

TOLERANCES = [(torch.float16, 1e-3), (torch.bfloat16, 1e-2), (torch.float32, 1e-6)]


@pytest.mark.parametrize("dtype,tol", TOLERANCES)
@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (4, 1000), (4, 4096), (0, 7)])
def test_matches_fp64_reference(dtype, tol, shape):
    rows, width = shape
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(width, generator=generator).to(dtype)
    weight = (torch.randn(shape, generator=generator) / math.sqrt(width)).to(dtype)
    before_x, before_weight = x.clone(), weight.clone()
    expected = torch.mv(weight.double(), x.double()).to(dtype)

    actual = gemv(x, weight)

    assert actual.shape == (rows,)
    assert actual.dtype == x.dtype and actual.device == x.device
    assert actual is not x and actual is not weight
    if rows:
        assert actual.data_ptr() not in (x.data_ptr(), weight.data_ptr())
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    torch.testing.assert_close(weight, before_weight, rtol=0, atol=0)


def test_matches_hand_calculated_example():
    actual = gemv(torch.tensor([1.0, 2.0]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    torch.testing.assert_close(actual, torch.tensor([11.0, 17.0]), rtol=0, atol=0)


def test_accumulates_in_fp32():
    x = torch.tensor([1000.0, 1000.0], dtype=torch.float16)
    weight = torch.tensor([[1000.0, -1000.0], [1.0, 1.0]], dtype=x.dtype)
    torch.testing.assert_close(gemv(x, weight), torch.tensor([0.0, 2000.0], dtype=x.dtype), rtol=0, atol=0)


@pytest.mark.parametrize(
    "device,amp_dtype",
    [
        ("cpu", torch.bfloat16),
        pytest.param("cuda", torch.float16,
                     marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")),
    ],
)
def test_preserves_fp32_under_autocast(device, amp_dtype):
    x = torch.full((2,), 1000.0, device=device)
    weight = torch.full((1, 2), 1000.0, device=device)

    with torch.autocast(device_type=device, dtype=amp_dtype):
        actual = gemv(x, weight)
        if device == "cuda":
            torch.testing.assert_close(gemv(x, weight, backend="triton"), actual, rtol=0, atol=0)
        assert torch.is_autocast_enabled(device)

    torch.testing.assert_close(actual, torch.tensor([2000000.0], device=device), rtol=0, atol=0)


@pytest.mark.parametrize("dtype,tol", TOLERANCES)
def test_supports_autograd(dtype, tol):
    x = torch.tensor([1.0, 2.0, -3.0], dtype=dtype, requires_grad=True)
    weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype, requires_grad=True)

    gemv(x, weight).sum().backward()

    torch.testing.assert_close(x.grad, weight.detach().sum(dim=0), rtol=tol, atol=tol)
    torch.testing.assert_close(weight.grad, x.detach().expand_as(weight), rtol=tol, atol=tol)


def test_empty_output_supports_autograd():
    x = torch.ones(5, requires_grad=True)
    weight = torch.empty(0, 5, requires_grad=True)

    gemv(x, weight).sum().backward()

    torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0, atol=0)
    assert weight.grad.shape == weight.shape


def test_accepts_contiguous_offset_views():
    x = torch.arange(6, dtype=torch.float32)[1:]
    weight = torch.arange(16, dtype=x.dtype)[1:].view(3, 5)
    before_x, before_weight = x.clone(), weight.clone()

    actual = gemv(x, weight)

    torch.testing.assert_close(actual, torch.mv(weight.double(), x.double()).float(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    torch.testing.assert_close(weight, before_weight, rtol=0, atol=0)


def test_accepts_overlapping_read_only_inputs():
    weight = torch.arange(15, dtype=torch.float32).view(3, 5)
    x = weight[0]
    before = weight.clone()

    actual = gemv(x, weight)

    torch.testing.assert_close(actual, torch.mv(before.double(), before[0].double()).float(), rtol=0, atol=0)
    torch.testing.assert_close(weight, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "x,weight,error,message",
    [
        ([1, 2], torch.ones(3, 2), TypeError, "torch.Tensor"),
        (torch.ones(2), [1, 2], TypeError, "torch.Tensor"),
        (torch.tensor(1.0), torch.ones(3, 1), ValueError, "nonempty vector"),
        (torch.ones(1, 5), torch.ones(3, 5), ValueError, "nonempty vector"),
        (torch.empty(0), torch.empty(3, 0), ValueError, "nonempty vector"),
        (torch.ones(5), torch.ones(5), ValueError, "weight must have shape"),
        (torch.ones(5), torch.ones(3, 4), ValueError, "weight must have shape"),
        (torch.ones(5, dtype=torch.float64), torch.ones(3, 5, dtype=torch.float64), TypeError, "float16"),
        (torch.ones(5), torch.ones(3, 5, dtype=torch.float16), ValueError, "dtype and device"),
        (torch.ones(5), torch.ones(3, 5, device="meta"), ValueError, "dtype and device"),
        (torch.ones(10)[::2], torch.ones(3, 5), ValueError, "contiguous"),
        (torch.ones(5), torch.ones(5, 3).t(), ValueError, "contiguous"),
    ],
    ids=["x-type", "weight-type", "scalar", "matrix-x", "empty-x", "weight-rank", "width",
         "dtype", "weight-dtype", "device", "strided-x", "strided-weight"],
)
@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_rejects_invalid_inputs(x, weight, error, message, backend):
    with pytest.raises(error, match=message):
        gemv(x, weight, backend=backend)


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be 'torch' or 'triton'"):
        gemv(torch.ones(2), torch.ones(3, 2), backend="unknown")


def test_triton_requires_cuda():
    with pytest.raises(ValueError, match="requires CUDA"):
        gemv(torch.ones(2), torch.ones(3, 2), backend="triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype,tol", TOLERANCES)
@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (4, 1000), (4, 4096), (3, 4097), (16, 8192), (128, 1024),
                                  (511, 1024), (512, 1023), (512, 1024), (512, 2048), (512, 2049), (0, 7)])
def test_triton_matches_reference(dtype, tol, shape):
    rows, width = shape
    generator = torch.Generator().manual_seed(0)
    x_cpu = torch.randn(width, generator=generator).to(dtype)
    weight_cpu = (torch.randn(shape, generator=generator) / math.sqrt(width)).to(dtype)
    x, weight = x_cpu.cuda(), weight_cpu.cuda()
    before_x, before_weight = x.clone(), weight.clone()
    expected64 = torch.mv(weight_cpu.double(), x_cpu.double()).to(dtype).cuda()

    actual = gemv(x, weight, backend="triton")

    assert actual.shape == (rows,) and actual.dtype == x.dtype and actual.device == x.device
    if rows:
        assert actual.data_ptr() not in (x.data_ptr(), weight.data_ptr())
    torch.testing.assert_close(actual, gemv(x, weight), rtol=tol, atol=tol)
    torch.testing.assert_close(actual, expected64, rtol=tol, atol=tol)
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    torch.testing.assert_close(weight, before_weight, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype,tol", TOLERANCES)
@pytest.mark.parametrize("shape", [(3, 5), (512, 1024)])
def test_triton_accepts_offset_and_read_only_aliasing(dtype, tol, shape):
    rows, width = shape
    generator = torch.Generator(device="cuda").manual_seed(0)
    offset_x = torch.randn(width + 1, device="cuda", dtype=dtype, generator=generator)[1:]
    storage = torch.randn(rows * width + 1, device="cuda", dtype=dtype, generator=generator) / math.sqrt(width)
    offset_weight = storage[1:].view(shape)
    for x, weight in ((offset_x, offset_weight), (offset_weight[0], offset_weight)):
        before_x, before_weight = x.clone(), weight.clone()
        torch.testing.assert_close(gemv(x, weight, backend="triton"), gemv(x, weight), rtol=tol, atol=tol)
        torch.testing.assert_close(x, before_x, rtol=0, atol=0)
        torch.testing.assert_close(weight, before_weight, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_accumulates_in_fp32():
    x = torch.tensor([1000.0, 1000.0], device="cuda", dtype=torch.float16)
    weight = torch.tensor([[1000.0, -1000.0], [1.0, 1.0]], device=x.device, dtype=x.dtype)
    torch.testing.assert_close(gemv(x, weight, backend="triton"), gemv(x, weight), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("input_index", [0, 1], ids=["x", "weight"])
def test_triton_rejects_autograd(input_index):
    inputs = [torch.ones(2, device="cuda"), torch.ones(3, 2, device="cuda")]
    inputs[input_index].requires_grad_()
    with pytest.raises(ValueError, match="does not support autograd"):
        gemv(*inputs, backend="triton")
