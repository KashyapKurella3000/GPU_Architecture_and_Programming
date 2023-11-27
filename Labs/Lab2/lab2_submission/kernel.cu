#include <stdio.h>
 
__global__ void vecAdd(const float *A, const float *B, float* C, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

    	if (idx < n) {
        	C[idx] = A[idx] + B[idx];
    	}
}
