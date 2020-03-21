import torch
import numpy
from modules.torch import *


def test_cupy_hamming_distance():
    for i in range(10):
        numpy_a = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)
        numpy_b = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)

        torch_a = torch.from_numpy(numpy_a).cuda()
        torch_b = torch.from_numpy(numpy_b).cuda()

        numpy_xor = numpy.bitwise_xor(numpy_a, numpy_b)
        numpy_dist = numpy.zeros((5, 5))
        for i in range(numpy_dist.shape[0]):
            for j in range(numpy_dist.shape[1]):
                numpy_dist[i, j] = numpy.binary_repr(numpy_xor[i, j]).count('1')

        torch_dist = int32_hamming_distance(torch_a, torch_b)
        numpy.testing.assert_equal(numpy_dist, torch_dist.cpu().numpy())
    print('Tested CFFI version.')


def test_cffi_hamming_distance():
    for i in range(10):
        numpy_a = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)
        numpy_b = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)

        torch_a = torch.from_numpy(numpy_a)
        torch_b = torch.from_numpy(numpy_b)

        numpy_xor = numpy.bitwise_xor(numpy_a, numpy_b)
        numpy_dist = numpy.zeros((5, 5))
        for i in range(numpy_dist.shape[0]):
            for j in range(numpy_dist.shape[1]):
                numpy_dist[i, j] = numpy.binary_repr(numpy_xor[i, j]).count('1')

        torch_dist = int32_hamming_distance(torch_a, torch_b)
        numpy.testing.assert_equal(numpy_dist, torch_dist.cpu().numpy())
    print('Tested CFFI version.')


def example():
    numpy_a = numpy.random.randint(0, 100, size=(1)).astype(numpy.int32)
    numpy_b = numpy.random.randint(0, 100, size=(1)).astype(numpy.int32)

    torch_a = torch.from_numpy(numpy_a).cuda()
    torch_b = torch.from_numpy(numpy_b).cuda()

    torch_dist = int32_hamming_distance(torch_a, torch_b)
    print('CuPy: %d and %d have hamming distance %d.' % (
        torch_a.item(),
        torch_b.item(),
        torch_dist.item(),
    ))

    torch_a = torch.from_numpy(numpy_a)
    torch_b = torch.from_numpy(numpy_b)

    torch_dist = int32_hamming_distance(torch_a, torch_b)
    print('CFFI: %d and %d have hamming distance %d.' % (
        torch_a.item(),
        torch_b.item(),
        torch_dist.item(),
    ))


if __name__ == '__main__':
    test_cupy_hamming_distance()
    test_cffi_hamming_distance()
    example()
