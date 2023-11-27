#include <stdio.h>
 
__global__ void VecAdd(const float* A, const float* B, float* C, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) C[tid] = A[tid] + B[tid];

}