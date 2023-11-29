%%cu
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]){
    float *A_h;
    float sum;
	int blockSize, gridSize, VecSize;

	FILE *fp = fopen("cpu-results.csv", "w");
	fprintf(fp,"Block, Grid, Size, Time\n");

    // Check all possibility
    for(gridSize = 1; gridSize <= 128; gridSize *= 2){
        for(blockSize = 32; blockSize <= 1024; blockSize *= 2){
            VecSize = gridSize * blockSize; // Calculate array size

            A_h = (float*) malloc( sizeof(float) * VecSize );
            for (int i=0; i < VecSize; i++) {
                A_h[i] = (float)i;
            }

            clock_t start = clock(); //Starts
            sum = 0;
            for (int i=0; i < VecSize; i++) {
                sum += A_h[i];
            }
            clock_t end = clock(); //Ends

            free(A_h);
            A_h = NULL;

            //Measure Data
            double elasped_secs = (((double) end - (double) start) / CLOCKS_PER_SEC) * 1000000;
            printf("Block: %d, Grid: %d, Size: %d, Time: %f us\n", blockSize, gridSize, VecSize, elasped_secs);
            fprintf(fp,"%d, %d, %d, %f\n", blockSize, gridSize, VecSize, elasped_secs);
        }
    }

	fclose(fp);

    return 0;
}
