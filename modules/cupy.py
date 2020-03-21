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

        dist[elem_idx] = 0;
        for(int byte_idx = 0; byte_idx < 4; byte_idx++) {
            unsigned int elem_byte_idx = 4*elem_idx+byte_idx;
            unsigned char a_byte = ((unsigned char*) a)[elem_byte_idx];
            unsigned char b_byte = ((unsigned char*) b)[elem_byte_idx];

            unsigned char d_byte = a_byte^b_byte;

            int d = 0;
            while(d_byte)
            {
              ++d;
              d_byte &= d_byte - 1; // why?
            }

            dist[elem_idx] += d;
        }
    }
'''