#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int size, rank;
    int localValue;
    int globalSum;
    int i;
    
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Each process calculates its local value
    localValue = (rank + 1) * 10;
    
    // Each process prints its local value
    printf("Process %d: Local Value = %d\n", rank, localValue);
    
    // Reduce all local values to a global sum at the root process (rank 0)
    MPI_Reduce(&localValue,   // Address of the local value to send
               &globalSum,    // Address to store the reduced result (only significant at root)
               1,             // Number of elements to reduce
               MPI_INT,       // Data type of elements
               MPI_SUM,       // Operation to perform (sum)
               0,             // Rank of the root process
               MPI_COMM_WORLD); // Communicator
    
    // Root process calculates and prints the results
    if (rank == 0) {
        printf("\nGlobal sum of all local values: %d\n", globalSum);
        
        // Calculate the expected sum for verification
        int expectedSum = 0;
        for (i = 1; i <= size; i++) {
            expectedSum += i * 10;
        }
        printf("Expected Sum: %d\n", expectedSum);
    }
    
    // Finalize MPI environment
    MPI_Finalize();
    
    return 0;
}