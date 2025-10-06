// Algorithm name is berkerely
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include <time.h>

#define MASTER_RANK 0

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Seed random number generator differently for each process
    srand(time(NULL) + rank);
    
    // Each process starts with its own local time (with some random offset)
    int local_time = 300 + (rand() % 100 - 50); // Base 300 with random offset [-50, 50]
    int master_time;
    
    printf("Process %d initial local time: %d\n", rank, local_time);
    
    // Master broadcasts its time to all processes (including itself)
    if (rank == MASTER_RANK) {
        master_time = local_time;
        printf("\nMaster broadcasting time: %d\n", master_time);
    }
    
    // Broadcast master's time to all processes (including master itself)
    MPI_Bcast(&master_time, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
    
    // Each process calculates its time difference from master's time
    int time_difference = local_time - master_time;
    printf("Process %d calculated difference: %d\n", rank, time_difference);
    
    // Use MPI_Reduce to sum all time differences at the master
    int sum_differences;
    MPI_Reduce(&time_difference, &sum_differences, 1, MPI_INT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
    
    // Master calculates average difference and new master time
    int average_difference;
    int new_master_time;
    int* adjustments = NULL;
    
    if (rank == MASTER_RANK) {
        average_difference = sum_differences / size;
        new_master_time = master_time + average_difference;
        
        printf("\nSum of differences: %d, Average difference: %d\n", sum_differences, average_difference);
        printf("New master time: %d\n", new_master_time);
        
        // Calculate adjustments for each process
        adjustments = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            adjustments[i] = new_master_time - (master_time + time_difference);
        }
    }
    
    // Master sends individual adjustments to each process using MPI_Send
    int my_adjustment;
    if (rank == MASTER_RANK) {
        // Master applies its own adjustment
        my_adjustment = adjustments[0];
        local_time += my_adjustment;
        printf("Master adjusting by %d to: %d\n", my_adjustment, local_time);
        
        // Send adjustments to all other processes
        for (int i = 1; i < size; i++) {
            MPI_Send(&adjustments[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            printf("Sent adjustment %d to process %d\n", adjustments[i], i);
        }
    } else {
        // Slaves receive their adjustment from master
        MPI_Recv(&my_adjustment, 1, MPI_INT, MASTER_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_time += my_adjustment;
        printf("Process %d adjusting by %d to: %d\n", rank, my_adjustment, local_time);
    }
    
    // Final synchronization check using MPI_Gather
    int* all_times = NULL;
    if (rank == MASTER_RANK) {
        all_times = (int*)malloc(size * sizeof(int));
    }
    
    MPI_Gather(&local_time, 1, MPI_INT, all_times, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
    
    if (rank == MASTER_RANK) {
        printf("\nFinal synchronized times:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: %d\n", i, all_times[i]);
        }
        free(all_times);
        free(adjustments);
    }
    
    MPI_Finalize();
    return 0;

}
