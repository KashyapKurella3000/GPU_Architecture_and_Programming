#include <time.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void VecAdd(const float *A, const float *B, float* C, long unsigned int n) {
	long unsigned int ID = blockIdx.x * blockDim.x + threadIdx.x;

    if (ID < n)
        C[ID] = A[ID] + B[ID];
}

int main (int argc, char *argv[]){
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
	int blockSize, gridSize;
	long unsigned int VecSize, Limit = (1 << 25); // Limit 4MB

	FILE *fp = fopen("results.csv", "w");
	fprintf(fp,"Method, Size, Time\n");

	printf("\nCudaMemcpy Test\n");
	for(VecSize = 8; VecSize <= Limit; VecSize *= 2){
		clock_t start = clock(); //Starts
		A_h = (float*) malloc( sizeof(float) * VecSize );
		B_h = (float*) malloc( sizeof(float) * VecSize );
		C_h = (float*) malloc( sizeof(float) * VecSize );
		
		for (long unsigned int i=0; i < VecSize; i++) {
			A_h[i] = 1.0f;
			B_h[i] = 2.0f;
		}

		cudaDeviceSynchronize();

		cudaMalloc(&A_d, sizeof(float) * VecSize);
		cudaMalloc(&B_d, sizeof(float) * VecSize);
		cudaMalloc(&C_d, sizeof(float) * VecSize);

		cudaMemcpy(A_d, A_h, sizeof(float) * VecSize, cudaMemcpyHostToDevice);
		cudaMemcpy(B_d, B_h, sizeof(float) * VecSize, cudaMemcpyHostToDevice);
		
		cudaDeviceSynchronize();

		blockSize = 32;
		gridSize = (int)ceil((float)VecSize/blockSize);
		
		VecAdd<<<gridSize, blockSize>>>(A_d, B_d, C_d, VecSize);

		cudaMemcpy(C_h, C_d, sizeof(float) * VecSize, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		free(A_h);
		free(B_h);
		free(C_h);

		cudaFree(A_d);
		cudaFree(B_d);
		cudaFree(C_d);

		cudaDeviceSynchronize();
		clock_t end = clock(); //Ends

		//Measure Data
		double elasped_secs = (((double) end - (double) start) / CLOCKS_PER_SEC) * 1000000;
		printf("Size: %ld, Time: %f us\n", sizeof(float) * VecSize, elasped_secs);
		fprintf(fp,"%s, %ld, %f\n", "CudaMemcpy", sizeof(float) * VecSize, elasped_secs);
	}

	printf("\nPinned Memory(cudaHostAlloc) Test\n");
	for(VecSize = 8; VecSize <= Limit; VecSize *= 2){
		clock_t start = clock(); //Starts
		cudaDeviceSynchronize();

		cudaHostAlloc(&A_h, sizeof(float) * VecSize, cudaHostAllocDefault);
		cudaHostAlloc(&B_h, sizeof(float) * VecSize, cudaHostAllocDefault);
		cudaHostAlloc(&C_h, sizeof(float) * VecSize, cudaHostAllocDefault);
		
		for (long unsigned int i=0; i < VecSize; i++) {
			A_h[i] = 1.0f;
			B_h[i] = 2.0f;
		}
		
		cudaHostGetDevicePointer(&A_d, A_h, 0);
		cudaHostGetDevicePointer(&B_d, B_h, 0);
		cudaHostGetDevicePointer(&C_d, C_h, 0);
		
		cudaDeviceSynchronize();

		blockSize = 32;
		gridSize = (int)ceil((float)VecSize/blockSize);
		
		VecAdd<<<gridSize, blockSize>>>(A_d, B_d, C_d, VecSize);

		cudaDeviceSynchronize();

		cudaFreeHost(A_h);
		cudaFreeHost(B_h);
		cudaFreeHost(C_h);

		A_d = NULL;
		B_d = NULL;
		C_d = NULL;

		cudaDeviceSynchronize();
		clock_t end = clock(); //Ends

		//Measure Data
		double elasped_secs = (((double) end - (double) start) / CLOCKS_PER_SEC) * 1000000;
		printf("Size: %ld, Time: %f us\n", sizeof(float) * VecSize, elasped_secs);
		fprintf(fp,"%s, %ld, %f\n", "cudaHostAlloc", sizeof(float) * VecSize, elasped_secs);
	}

	printf("\nUnified Virtual Memory(cudaMallocManaged) Test\n");
	for(VecSize = 8; VecSize <= Limit; VecSize *= 2){
		clock_t start = clock(); //Starts
		cudaDeviceSynchronize();

		cudaMallocManaged(&A_h, sizeof(float) * VecSize);
		cudaMallocManaged(&B_h, sizeof(float) * VecSize);
		cudaMallocManaged(&C_h, sizeof(float) * VecSize);
		
		for (long unsigned int i=0; i < VecSize; i++) {
			A_h[i] = 1.0f;
			B_h[i] = 2.0f;
		}
		
		cudaDeviceSynchronize();

		blockSize = 32;
		gridSize = (int)ceil((float)VecSize/blockSize);
		
		VecAdd<<<gridSize, blockSize>>>(A_h, B_h, C_h, VecSize);

		cudaDeviceSynchronize();

		cudaFree(A_h);
		cudaFree(B_h);
		cudaFree(C_h);

		cudaDeviceSynchronize();
		clock_t end = clock(); //Ends

		//Measure Data
		double elasped_secs = (((double) end - (double) start) / CLOCKS_PER_SEC) * 1000000;
		printf("Size: %ld, Time: %f us\n", sizeof(float) * VecSize, elasped_secs);
		fprintf(fp,"%s, %ld, %f\n", "cudaMallocManaged", sizeof(float) * VecSize, elasped_secs);
	}

	fclose(fp);

    return 0;
}
