#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char** argv) {
    int rank, size;
    int bcast_data;
    int local_val, sum_result;
    int *send_array = NULL;  // Pointer for scatter send buffer
    int scatter_val;
    int *gather_array = NULL; // Pointer for gather receive buffer
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        bcast_data = 100;
        printf("P0 broadcasting: %d\n", bcast_data);
    }
    MPI_Bcast(&bcast_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("P%d got broadcast: %d\n", rank, bcast_data);

    // Allocate send_array only on root for scatter
    if (rank == 0) {
        send_array = (int*)malloc(size * sizeof(int));
        for (i = 0; i < size; i++) {
            send_array[i] = (i + 1) * 10;
        }
        printf("\nP0 scattering: ");
        for (i = 0; i < size; i++) printf("%d ", send_array[i]);
        printf("\n");
    }

    MPI_Scatter(send_array, 1, MPI_INT, &scatter_val, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("P%d got scattered: %d\n", rank, scatter_val);

    // Free send_array on root after scatter
    if (rank == 0) {
        free(send_array);
    }

    local_val = (rank + 1) * 5;
    printf("\nP%d local value: %d\n", rank, local_val);
    MPI_Reduce(&local_val, &sum_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("P0 reduce sum: %d\n", sum_result);
    }

    // Allocate gather_array only on root for gather
    if (rank == 0) {
        gather_array = (int*)malloc(size * sizeof(int));
    }

    int gather_val = rank * 2;
    MPI_Gather(&gather_val, 1, MPI_INT, gather_array, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\nP0 gathered: ");
        for (i = 0; i < size; i++) printf("%d ", gather_array[i]);
        printf("\n");
        free(gather_array);
    }

    MPI_Finalize();
    return 0;
}