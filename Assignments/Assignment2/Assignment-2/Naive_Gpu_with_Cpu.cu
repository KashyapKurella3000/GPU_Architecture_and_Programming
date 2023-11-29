%%cu
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono> // For CPU timing
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Function to initialize a matrix with random values
void initMatrix(float *matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

// CPU Matrix Multiplication
void matrixMulCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Function to display a matrix to the console
void displayMatrix(float *matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}
__global__ void matrixMulNaive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {

    int N = 4; // Matrix size
    size_t matrix_size = N * N * sizeof(float);
    
    //Max Shared Memory per block
    int maxSharedMemoryPerBlock;
    cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlock,0);
    std::cout << "Max Shared Memory Per Block: " << maxSharedMemoryPerBlock << " bytes" << std::endl;

	// Find and print the maximum threads per block
    int maxThreadsPerBlock;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Assuming device 0
    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    std::cout << "Max Threads Per Block: " << maxThreadsPerBlock << std::endl;

    
    // Allocate memory on host and initialize matrices with random values
    float *h_A = new float[matrix_size];
    float *h_B = new float[matrix_size];
    float *h_C_cpu = new float[matrix_size];
    float *h_C_gpu = new float[matrix_size];

    srand(static_cast<unsigned int>(time(nullptr)));
    initMatrix(h_A, N);
    initMatrix(h_B, N);
    
    
     // Create CPU timers for CPU timing
    std::chrono::high_resolution_clock::time_point cpu_start, cpu_stop;
    
	 // Start measuring CPU time
    cpu_start = std::chrono::high_resolution_clock::now();

    // Perform CPU matrix multiplication
    matrixMulCPU(h_A, h_B, h_C_cpu, N);
    
    // Stop measuring CPU time
    cpu_stop = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_stop - cpu_start);
    float cpu_seconds = cpu_duration.count() ; // Convert to seconds
    std::cout << "CPU Execution Time: " << cpu_seconds << "ms" << std::endl;

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrix_size);
    cudaMalloc((void**)&d_B, matrix_size);
    cudaMalloc((void**)&d_C, matrix_size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);

    // Define thread block and grid dimensions

    /* dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y); */

    // Define thread block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE); //Block Dimensions
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE); //Grid Dimensions
    
    // Create CUDA events for GPU timing
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    // Start measuring GPU time
    cudaEventRecord(gpu_start);

    // Launch the kernel  
    matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    
    // Stop measuring GPU time
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);

    // Calculate and print GPU execution time in seconds
    float gpu_milliseconds = 0.0f;
    cudaEventElapsedTime(&gpu_milliseconds, gpu_start, gpu_stop);
    float gpu_seconds = gpu_milliseconds ; // Convert to seconds
    std::cout << "GPU Execution Time: " << gpu_seconds << "ms" << std::endl;

    // Copy the result back to the host
    cudaMemcpy(h_C_gpu, d_C, matrix_size, cudaMemcpyDeviceToHost);

    // Display the output matrix
    /*std::cout << " GPU Output Matrix:" << std::endl;
    displayMatrix(h_C_gpu, N);

    std::cout << " \n------------End of GPU Matrix----------\n" << std::endl;

    std::cout << " CPU Output Matrix:" << std::endl;
    displayMatrix(h_C_gpu, N);

    std::cout << "\n ------------End of CPU Matrix---------- \n" << std::endl; */

    // Compare GPU and CPU results for verification
    
    bool verificationPassed = true;
    for(int i =0; i < N * N; ++i) {
        if(fabs(h_C_gpu[i] - h_C_cpu[i]) > 1e-5){
            std::cout << "\n Verification Failed!! At " << i <<" "<<h_C_gpu[i] << " " <<h_C_cpu[i] << std::endl;
            verificationPassed = false;
            break;
        } 
    }
    if(verificationPassed){
        std::cout << "\n Verification Sucess!! \n" <<std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;

    return 0;
}
