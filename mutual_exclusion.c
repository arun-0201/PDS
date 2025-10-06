#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // For sleep()

// Message Tags
#define REQUEST 10
#define REPLY   11

// Process States
#define RELEASED 0
#define WANTED   1
#define HELD     2

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // State variables
    int my_state = RELEASED;
    int my_ts = 0;        // Lamport timestamp
    int request_ts = 0;   // Timestamp of our request
    int replies_received = 0;

    // Queue for deferred requests (process ranks)
    int *deferred_queue = (int*)malloc(world_size * sizeof(int));
    int deferred_count = 0;
    int i;
    // --- 1. Request to Enter Critical Section ---
    my_state = WANTED;
    my_ts++;
    request_ts = my_ts;

    printf("Process %d requesting CS (ts=%d)\n", my_rank, request_ts);
    fflush(stdout);

    // Broadcast request to all other processes
    int request_msg[2] = {request_ts, my_rank};

    for (i = 0; i < world_size; i++) {
        if (i != my_rank) {
            MPI_Send(request_msg, 2, MPI_INT, i, REQUEST, MPI_COMM_WORLD);
        }
    }

    // --- 2. Wait for Replies and Handle Incoming Messages ---
    while (replies_received < world_size - 1) {
        MPI_Status status;
        int recv_msg[2]; // [0]=timestamp, [1]=rank

        // Block and wait for any incoming message
        MPI_Recv(recv_msg, 2, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	// Update our logical clock
        my_ts = (recv_msg[0] > my_ts) ? recv_msg[0] + 1 : my_ts + 1;

        if (status.MPI_TAG == REQUEST) {
            int their_rank = status.MPI_SOURCE;
            int their_ts = recv_msg[0];
            printf("Process %d received REQUEST from %d (ts=%d)\n", my_rank, their_rank, their_ts);
            fflush(stdout);

            // Priority check: lower timestamp wins, lower rank is the tie-breaker
            int they_have_priority = (their_ts < request_ts) || (their_ts == request_ts && their_rank < my_rank);

            if (my_state == HELD || (my_state == WANTED && !they_have_priority)) {
                // Defer reply
                deferred_queue[deferred_count++] = their_rank;
            } else {
                // Grant reply
                MPI_Send(NULL, 0, MPI_INT, their_rank, REPLY, MPI_COMM_WORLD);
            }
        }
        else if (status.MPI_TAG == REPLY) {
            printf("Process %d received OK from %d\n", my_rank, status.MPI_SOURCE);
            fflush(stdout);
            replies_received++;
        }
    }

    // --- 3. Enter and Exit Critical Section ---
    my_state = HELD;
    printf("\n>>> Process %d in critical section <<<\n\n", my_rank);
    fflush(stdout);
    sleep(2); // Simulate work

    printf("Process %d releasing CS\n", my_rank);
    fflush(stdout);
    my_state = RELEASED;

    // --- 4. Send Replies to Deferred Processes ---
    for (i = 0; i < deferred_count; i++) {
        MPI_Send(NULL, 0, MPI_INT, deferred_queue[i], REPLY, MPI_COMM_WORLD);
    }

    // Wait for all processes to finish before exiting
    MPI_Barrier(MPI_COMM_WORLD);
    free(deferred_queue);
    MPI_Finalize();
    return 0;
}
