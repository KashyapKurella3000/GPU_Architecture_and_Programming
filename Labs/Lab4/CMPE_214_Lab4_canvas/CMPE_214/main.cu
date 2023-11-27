#include <stdio.h>
#include <stdlib.h>

__device__ void P_chasing2(int *A, int *I, long long int *T, long long int iterations, int starting_index){    
	__shared__ long long int s_tvalue[1024 * 4];
    __shared__ int s_index[1024 * 4];
    int j = starting_index;

    long long int start_time = 0;
    long long int end_time = 0;
    long long int time_interval = 0;

    asm(".reg .u64 t1;\n\t"
    	".reg .u64 t2;\n\t");

    for (long long int it = 0; it < iterations; it++){
		asm("mul.wide.u32 t1, %2, %4;\n\t"
			"add.u64 t2, t1, %3;\n\t"
			"mov.u64 %0, %clock64;\n\t"
			"ld.global.u32 %1, [t2];\n\t"
			: "=l"(start_time), "=r"(j) : "r"(j), "l"(A), "r"(4));

		s_index[it] = j;

		asm volatile ("mov.u64 %0, %clock64;": "=l"(end_time));

		time_interval = end_time - start_time;
		s_tvalue[it] = time_interval;
	}

	for (long long int it = 0; it < iterations; it++){
		I[it] = s_index[it];
		T[it] = s_tvalue[it];
	}
}
// to execute kernel
__global__ void kernel(int *A, int *I, long long int *T, long long int iterations, int starting_index){
	P_chasing2(A, I, T, iterations, starting_index);
}

void init_cpu_data(int* A, long long int size, int stride, long long int mod){
	for (long long int i = 0; i < size; i = i + stride){
    	A[i]=(i + stride) % mod;
	}
}

int main (int argc, char *argv[]){
	int stride = 1;
    int *A_h, *A_d, *I_h, *I_d;
	long long int *T_h, *T_d;
    long long int N = 4096, mod = 4096;
	
    A_h = (int*) malloc( sizeof(int) * N );
	I_h = (int*) malloc( sizeof(int) * N );
	T_h = (long long int*) malloc ( sizeof(long long int) * N );
	init_cpu_data(A_h, N, stride, mod);

    cudaDeviceSynchronize();

	cudaMalloc(&A_d, sizeof(int) * N);
	cudaMalloc(&I_d, sizeof(int) * N);
	cudaMalloc(&T_d, sizeof(long long int) * N);
	cudaMemcpy(A_d, A_h, sizeof(int) * N, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    kernel<<<1, 1>>>(A_d, I_d, T_d, N, 0);
	cudaMemcpy(I_h, I_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(T_h, T_d, sizeof(long long int) * N, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

	FILE *fp = fopen("results.csv", "w");
	fprintf(fp,"A, Index, Time\n");

	for (long long int it = 0; it < N; it++){
		fprintf(fp,"%d, %d, %lld\n", A_h[it], I_h[it], T_h[it]);
		printf("A: %d, I: %d, Time: %lld\n", A_h[it], I_h[it], T_h[it]);
	}

	fclose(fp);

    free(A_h);
	free(I_h);
	free(T_h);
    cudaFree(A_d);
	cudaFree(I_d);
    cudaFree(T_d);

    return 0;
}
