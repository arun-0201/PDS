#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define N 8

__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
    int tid = threadIdx.x;
    
    if (tid < N)
        d_c[tid] = d_a[tid] * d_b[tid];
    
    __syncthreads();
    
    for (int s = 1; s < N; s *= 2) {
        if (tid % 2 == 0) {
            d_c[tid] = d_c[tid] + d_c[tid + s];
        }
        __syncthreads();
    }
}

int main(void) {
    int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;
    
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));
    
    for (int i = 0; i < N; i++) {
        h_a[i] = 1;
        h_b[i] = 1;
    }
    
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 
    float milliseconds = 0; 

    cudaEventRecord(start); 
    
    gpuAdd<<<1, N>>>(d_a, d_b, d_c);

    cudaEventRecord(stop); 
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop); 
    
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Vector Dot Product! \n");
    for (int i = 0; i < N; i++) {
        printf("The array value at index %d is %d\n", i, h_c[i]);
    }
    
    printf("Dot Product is %d\n", h_c[0]);
    printf("Kernel execution time: %f milliseconds\n", milliseconds);

    cudaEventDestroy(start); 
    cudaEventDestroy(stop);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;

}
