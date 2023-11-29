#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 64

// Function to generate random integers in a given range
int randomInt(int min, int max) {
    return min + rand() % (max - min + 1);
}

// Function to multiply two matrices
void multiplyMatrices(int **matA, int **matB, int **result, int size) {
    for (int i = 0; i < size; i += BLOCK_SIZE) {
        for (int j = 0; j < size; j += BLOCK_SIZE) {
            for (int k = 0; k < size; k += BLOCK_SIZE) {
                for (int ii = i; ii < i + BLOCK_SIZE && ii < size; ii++) {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < size; jj++) {
                        for (int kk = k; kk < k + BLOCK_SIZE && kk < size; kk++) {
                            result[ii][jj] += matA[ii][kk] * matB[kk][jj];
                        }
                    }
                }
            }
        }
    }
}

// Function to display a matrix
void displayMatrix(int **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    srand(time(NULL)); // Seed the random number generator

    int size; // Size of the matrices (2, 4, 8, ..., 8192)
    printf("Enter the size of matrices (2, 4, 8, ..., 8192): ");
    scanf("%d", &size);

    // Allocate memory for matrices
    int **matrixA = (int **)malloc(size * sizeof(int *));
    int **matrixB = (int **)malloc(size * sizeof(int *));
    int **result = (int **)malloc(size * sizeof(int *));

    for (int i = 0; i < size; i++) {
        matrixA[i] = (int *)malloc(size * sizeof(int));
        matrixB[i] = (int *)malloc(size * sizeof(int));
        result[i] = (int *)malloc(size * sizeof(int));
    }

    // Initialize matrices with random integers
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrixA[i][j] = randomInt(1, 10);
            matrixB[i][j] = randomInt(1, 10);
        }
    }

    // Measure execution time
    clock_t start = clock();
    multiplyMatrices(matrixA, matrixB, result, size);
    clock_t end = clock();

    // Display result matrix
    printf("Result Matrix:\n");
    displayMatrix(result, size);

    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Matrix multiplication took %.2f seconds.\n", elapsed_time);

    // Free allocated memory
    for (int i = 0; i < size; i++) {
        free(matrixA[i]);
        free(matrixB[i]);
        free(result[i]);
    }
    free(matrixA);
    free(matrixB);
    free(result);

    return 0;
}
