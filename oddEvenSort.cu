#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void oddEvenSortKernel(int *dev_array, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Each thread handles one element and participates in all phases
    for (int phase = 0; phase < n; phase++) {
        __syncthreads(); // Synchronize after each phase
        
        if (phase % 2 == 0) {
            // Even phase: even-indexed threads compare with their right neighbor
            if (idx % 2 == 0 && idx < n - 1) {
                if (dev_array[idx] > dev_array[idx + 1]) {
                    // Swap elements
                    int temp = dev_array[idx];
                    dev_array[idx] = dev_array[idx + 1];
                    dev_array[idx + 1] = temp;
                }
            }
        } else {
            // Odd phase: odd-indexed threads compare with their right neighbor
            if (idx % 2 == 1 && idx < n - 1) {
                if (dev_array[idx] > dev_array[idx + 1]) {
                    // Swap elements
                    int temp = dev_array[idx];
                    dev_array[idx] = dev_array[idx + 1];
                    dev_array[idx + 1] = temp;
                }
            }
        }
    }
}

int main() {
    const int n = 16;
    int h_array[n] = {5, 2, 9, 1, 5, 6, 3, 8, 4, 7, 0, 11, 14, 13, 12, 10};
    int *d_array;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // Print original array
    printf("Original array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n\n");
    
    // Allocate device memory
    cudaMalloc((void**)&d_array, n * sizeof(int));
    
    // Copy array to device
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Record start event
    cudaEventRecord(start);
    
    // Launch the sorting kernel (single kernel call)
    oddEvenSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, n);
    
    // Record stop event
    cudaEventRecord(stop);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy sorted array back to host
    cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print sorted array
    printf("Sorted array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n\n");
    
    // Print execution time
    printf("Kernel execution time: %.3f milliseconds\n", milliseconds);
    printf("Array size: %d elements\n", n);
    
    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Free device memory
    cudaFree(d_array);
    
    return 0;
}
