#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to print an array
void print_array(int *arr, int size, const char *title) {
    printf("%s: [", title);
    int i; // C89 style: declaration at top of block
    for (i = 0; i < size; i++) {
        printf("%d", arr[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

int main(int argc, char* argv[]) {
    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // N is the size of the array, which must equal the number of processes
    const int N = world_size;

    int *initial_array = NULL;
    int *sorted_array = NULL;
    int i; // C89 style: declare loop variable for root process block

    // --- Step 1: Initialization (Root Process Only) ---
    if (world_rank == 0) {
        if (N <= 1) {
            fprintf(stderr, "This program requires at least 2 processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Running Enumeration Sort with N = %d processes.\n", N);

        initial_array = (int*)malloc(N * sizeof(int));
        sorted_array = (int*)malloc(N * sizeof(int));

        // Seed random number generator and fill the array
        srand(time(NULL));
        for (i = 0; i < N; i++) {
            initial_array[i] = rand() % 100; // Random numbers between 0 and 99
        }

        print_array(initial_array, N, "Initial Unsorted Array");
    }

    // --- Step 2: Distribution (Scatter) ---
    int my_value;
    MPI_Scatter(
        initial_array,
        1, MPI_INT,
        &my_value,
        1, MPI_INT,
        0, MPI_COMM_WORLD
    );

    // --- Step 3: Data Sharing (Allgather) ---
    // Each process needs all other values to perform the comparison
    // ERROR FIX: 'all_values' must be an integer pointer (int*) and cast must be (int*)
    int *all_values = (int*)malloc(N * sizeof(int));
    MPI_Allgather(
        &my_value,
        1, MPI_INT,
        all_values, // This now correctly passes a pointer
        1, MPI_INT,
        MPI_COMM_WORLD
    );

    // --- Step 4: Local Rank Calculation ---
    int my_rank_in_sorted = 0;
    int j; // C89 style: declaration at top of block
    for (j = 0; j < N; j++) {
        if (all_values[j] < my_value) {
            my_rank_in_sorted++;
        }
        else if (all_values[j] == my_value && j < world_rank) {
            my_rank_in_sorted++;
        }
    }

    // Free the all_values array as it's no longer needed
    free(all_values);

    // --- Step 5 & 6: Result Collection and Finalization ---
    if (world_rank == 0) {
        int k; // C89 style: declaration at top of block
        int received_value;
        MPI_Status status;
        
        // The root process places its own value directly
        sorted_array[my_rank_in_sorted] = my_value;

        // Receive values from all other processes
        for (k = 1; k < N; k++) {
            MPI_Recv(
                &received_value, 1, MPI_INT,
                MPI_ANY_SOURCE, MPI_ANY_TAG,
                MPI_COMM_WORLD, &status
            );

            int source_rank_in_sorted = status.MPI_TAG;
            sorted_array[source_rank_in_sorted] = received_value;
        }

        print_array(sorted_array, N, "Final Sorted Array");

        free(initial_array);
        free(sorted_array);

    } else {
        // All non-root processes send their value to the root
        MPI_Send(
            &my_value, 1, MPI_INT,
            0, my_rank_in_sorted,
            MPI_COMM_WORLD
        );
    }

    MPI_Finalize();
    return 0;
}