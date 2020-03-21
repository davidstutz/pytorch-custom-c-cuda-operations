import torch
from . import cffi
from . import cupy


def is_cuda(mixed):
    """
    Check if model/tensor is on CUDA.

    :param mixed: model or tensor
    :type mixed: torch.nn.Module or torch.autograd.Variable or torch.Tensor
    :return: on cuda
    :rtype: bool
    """

    assert isinstance(mixed, torch.nn.Module) or isinstance(mixed, torch.autograd.Variable) \
        or isinstance(mixed, torch.Tensor), 'mixed has to be torch.nn.Module, torch.autograd.Variable or torch.Tensor'

    is_cuda = False
    if isinstance(mixed, torch.nn.Module):
        is_cuda = True
        for parameters in list(mixed.parameters()):
            is_cuda = is_cuda and parameters.is_cuda
    if isinstance(mixed, torch.autograd.Variable):
        is_cuda = mixed.is_cuda
    if isinstance(mixed, torch.Tensor):
        is_cuda = mixed.is_cuda

    return is_cuda


def int32_hamming_distance(a, b):
    """
    Bit-wise hamming distance.

    :param a: first tensor
    :type a: torch.Tensor
    :param b: first tensor
    :type b: torch.Tensor
    :return: hamming distance
    :rtype: torch.Tensor
    """

    #assert (a.is_contiguous() == True)
    if not a.is_contiguous():
        a.contiguous()
    assert (a.dtype == torch.int32)
    cuda = is_cuda(a)

    #assert (b.is_contiguous() == True)
    if not b.is_contiguous():
        b.contiguous()
    assert (b.dtype == torch.int32)
    assert is_cuda(b) is cuda

    assert len(a.shape) == len(a.shape)
    for d in range(len(a.shape)):
        assert a.shape[d] == b.shape[d]

    dist = a.new_zeros(a.shape).int()
    n = dist.nelement()

    if cuda:
        cupy.cunnex('cupy_int32hammingdistance')(
            grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
            block=tuple([512, 1, 1]),
            args=[n,
                  a.data_ptr(),
                  b.data_ptr(),
                  dist.data_ptr()],
            stream=cupy.Stream
        )
    else:
        _n = cffi.ffi.cast('int', n)
        _a = cffi.ffi.cast('int*', a.data_ptr())
        _b = cffi.ffi.cast('int*', b.data_ptr())
        _dist = cffi.ffi.cast('int*', dist.data_ptr())

        cffi.lib.cffi_int32hammingdistance(_n, _a, _b, _dist)

    return dist