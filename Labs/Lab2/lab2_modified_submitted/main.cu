#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"

int main (int argc, char *argv[]){

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;    
    unsigned VecSize;
   
    if (argc == 1) {
        VecSize = 256;
    } else if (argc == 2) {
      VecSize = atoi(argv[1]);
    } else {
        printf("Usage: ./vecAdd <Size>");
        exit(0);
    }
	
    A_h = (float*) malloc( sizeof(float) * VecSize );
	B_h = (float*) malloc( sizeof(float) * VecSize );
	C_h = (float*) malloc( sizeof(float) * VecSize );

    cudaMalloc((void**)&A_d, VecSize);
    cudaMalloc((void**)&B_d, VecSize);
    cudaMalloc((void**)&C_d, VecSize);

	
    for (unsigned int i=0; i < VecSize; i++) {
		A_h[i] = i;
		B_h[i] = i;
	}    

    cudaDeviceSynchronize();
    
    //INSERT Memory CODE HERE
    cudaMemcpy(A_d,A_h,sizeof(float)*VecSize, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,sizeof(float)*VecSize, cudaMemcpyHostToDevice);
    

    cudaDeviceSynchronize();

    dim3 blockDims(256);
    dim3 gridDims = ((VecSize + blockDims.x -1)/ blockDims.x);

    //INSERT kernel launch CODE HERE
    VecAdd<<<gridDims,blockDims>>>(A_d,B_d,C_d,VecSize);

    cudaDeviceSynchronize();

    cudaMemcpy(C_h,C_d, sizeof(float)*VecSize,cudaMemcpyDeviceToHost);

    for(int l =0; l<8;l++){
        printf("%f", C_h[l]);
    }

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT Memory CODE HERE
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
