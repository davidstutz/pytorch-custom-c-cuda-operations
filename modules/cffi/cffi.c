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
}