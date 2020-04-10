#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>

void cffi_int32hammingdistance(
    const int n,
    const int* a,
    const int* b,
    int* dist
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        dist[elem_idx] = 0;
        int x = a[elem_idx] ^ b[elem_idx];

        while(x != 0) {
            x = x & (x-1);
            dist[elem_idx]++;
        }
    }
}