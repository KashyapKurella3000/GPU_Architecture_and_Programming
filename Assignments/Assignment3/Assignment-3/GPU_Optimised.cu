%%cu
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define SHMEM_SIZE 1024

__global__ void SumReduction(float *A, float *B) {
	// Allocate shared memory
	__shared__ float partial_sum[SHMEM_SIZE];

    // Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = A[tid];

    // Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();

		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s)
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
	}

    __syncthreads();

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		B[blockIdx.x] = partial_sum[0];
	}
}

int main (int argc, char *argv[]){
    float *A_h, *A_d;
    float *sum_h, *sum_d;
	int blockSize, gridSize, VecSize;

	FILE *fp = fopen("gpu-opti-results.csv", "w");
	fprintf(fp,"Block, Grid, Size, Time\n");

    // Check all possibility
    for(gridSize = 1; gridSize <= 128; gridSize *= 2){
        for(blockSize = 32; blockSize <= 1024; blockSize *= 2){
            VecSize = gridSize * blockSize; // Calculate array size

            A_h = (float*) malloc( sizeof(float) * VecSize );
            sum_h = (float*) malloc( sizeof(float) * VecSize );
            for (int i=0; i < VecSize; i++) {
                A_h[i] = (float)i;
            }

            cudaDeviceSynchronize();

		    cudaMalloc(&A_d, sizeof(float) * VecSize);
            cudaMalloc(&sum_d, sizeof(float) * VecSize);

            cudaMemcpy(A_d, A_h, sizeof(float) * VecSize, cudaMemcpyHostToDevice);
		
		    cudaDeviceSynchronize();

            clock_t start = clock(); //Starts
            
            SumReduction<<<gridSize, blockSize>>>(A_d, sum_d);
            SumReduction<<<1, blockSize>>>(A_d, sum_d);

            clock_t end = clock(); //Ends

            cudaMemcpy(sum_h, sum_d, sizeof(float) * VecSize, cudaMemcpyDeviceToHost);

            cudaDeviceSynchronize();

            free(A_h);
            free(sum_h);

            cudaFree(A_d);
            cudaFree(sum_d);

            A_h = NULL;
            A_d = NULL;
            sum_h = NULL;
            sum_d = NULL;

            cudaDeviceSynchronize();

            //Measure Data
            double elasped_secs = (((double) end - (double) start) / CLOCKS_PER_SEC) * 1000000;
            printf("Block: %d, Grid: %d, Size: %d, Time: %f us\n", blockSize, gridSize, VecSize, elasped_secs);
            fprintf(fp,"%d, %d, %d, %f\n", blockSize, gridSize, VecSize, elasped_secs);
        }
    }

	fclose(fp);

    return 0;
}
