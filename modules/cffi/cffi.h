/**
 * Bitwise hamming distance.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param dist distance tensor
 */
void cffi_int32hammingdistance(
    const int n,
    const int* a,
    const int* b,
    int* dist
);