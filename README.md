# Custom Tensor Operations in PyTorch using C and CUDA

This repository contains a simple example of implementing custom tensor operations
for [PyTorch](https://pytorch.org/) in C/CUDA using [CFFI](https://cffi.readthedocs.io/en/latest/)
and [CuPy](https://cupy.chainer.org/).

Blog article: [Implementing Custom PyTorch Tensor Operations in C and CUDA](https://davidstutz.de/implementing-custom-pytorch-tensor-operations-in-c-and-cuda/)

## Installation

Requirements:

* CUDA
* PyTorch
* CFFI (`pip install cffi`)
* CuPy (see [here](https://docs-cupy.chainer.org/en/stable/install.html) for installation instructions for various CUDA version)

No compilation is required. Compilation of the CFFI code in `cffi/` is done on-the-fly.

## License

Copyright (c) 2020, David Stutz All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
* Neither the name of David Stutz nor the names of its contributors may be
used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.
