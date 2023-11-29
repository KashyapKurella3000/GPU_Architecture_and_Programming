#include <iostream>
#include <cstdlib>
#include <ctime>

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
__global__ void matrixMulTiled(float *A, float *B, float *C, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int i = 0; i < N / TILE_SIZE; ++i) {
        shared_A[ty][tx] = A[row * N + (i * TILE_SIZE + tx)];
        shared_B[ty][tx] = B[(i * TILE_SIZE + ty) * N + col];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}


int main() {
    int N = 16; // Matrix size (power of 2)
    size_t matrix_size = N * N * sizeof(float);

    // Allocate memory on host and initialize matrices with random values
    float *h_A = new float[matrix_size];
    float *h_B = new float[matrix_size];
    float *h_C_cpu = new float[matrix_size];
    float *h_C_gpu = new float[matrix_size];

    srand(static_cast<unsigned int>(time(nullptr)));
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Perform CPU matrix multiplication
    matrixMulCPU(h_A, h_B, h_C_cpu, N);

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrix_size);
    cudaMalloc((void**)&d_B, matrix_size);
    cudaMalloc((void**)&d_C, matrix_size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);

    // Define thread block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy the result back to the host
    cudaMemcpy(h_C_gpu, d_C, matrix_size, cudaMemcpyDeviceToHost);

    // Display the output matrix
    std::cout << " GPU Output Matrix:" << std::endl;
    displayMatrix(h_C_gpu, N);

    std::cout << " \n------------End of GPU Matrix----------\n" << std::endl;

    std::cout << " CPU Output Matrix:" << std::endl;
    displayMatrix(h_C_gpu, N);

    std::cout << "\n ------------End of CPU Matrix---------- \n" << std::endl;

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
