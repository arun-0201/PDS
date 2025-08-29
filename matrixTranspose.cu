#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define N 9
#define WIDTH 3

__global__ void transpose(int *d_m, int *d_t)
{
    int tid = threadIdx.x;
    int row = tid / WIDTH;
    int col = tid % WIDTH;
    
    int transposed_index = col * WIDTH + row;
    d_t[transposed_index] = d_m[tid];
}

int main(void)
{
    int a[N], b[N];
    int *d_m, *d_t;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    cudaMalloc((void**) &d_m, N * sizeof(int));
    cudaMalloc((void**) &d_t, N * sizeof(int));

    for (int i=0; i<N; i++)
    {
        a[i] = i+1;
    }

    cudaMemcpy(d_m, a, N*sizeof(int), cudaMemcpyHostToDevice);

    // Record start event
    cudaEventRecord(start);
    
    transpose<<<1, N>>>(d_m, d_t);
    
    // Record stop event
    cudaEventRecord(stop);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(b, d_t, N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Original matrix (3x3):\n");
    for (int i=0; i<N; i++)
    {
        printf("%d\t", a[i]);
        if ((i+1) % WIDTH == 0) printf("\n");
    }

    printf("\nTranspose of the matrix (3x3):\n");
    for (int i=0; i<N; i++)
    {
        printf("%d\t", b[i]);
        if ((i+1) % WIDTH == 0) printf("\n");
    }
    
    // Print execution time
    printf("\nKernel execution time: %.3f milliseconds\n", milliseconds);
    printf("Matrix size: %dx%d (%d elements)\n", WIDTH, WIDTH, N);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_m);
    cudaFree(d_t);

    return 0;
}
