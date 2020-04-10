import torch


try:
    import cupy
    @cupy.util.memoize(for_each_device=True)
    def cunnex(strFunction):
        return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)
except ImportError:
    print("CUPY cannot initialize, not using CUDA kernels")


class Stream:
    ptr = torch.cuda.current_stream().cuda_stream


cupy_int32hammingdistance = '''
    extern "C" __global__ void cupy_int32hammingdistance(
        const int n,
        const int* a,
        const int* b,
        int* dist
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (elem_idx >= n) {
            return;
        }

        int x = a[elem_idx] ^ b[elem_idx];

        while(x != 0) {
            x = x & (x-1);
            dist[elem_idx]++;
        }
    }
'''