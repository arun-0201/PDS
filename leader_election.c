#include <mpi.h>
#include <stdio.h>
#include <unistd.h>  // for sleep(), optional for pacing output

int main(int argc, char *argv[]) {
    int rank, size;
    int token;              // carries the max rank seen so far in election
    int elected_leader;
    int initiator = 1;      // Simulated failure detector: Process 1 starts election
    int failed_leader;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    failed_leader = size - 1;  // Assume highest rank was leader and has "failed"

    if (rank == initiator) {
        printf(">>> [Process %d] Detected failure of current coordinator (Process %d). Initiating ELECTION...\n", rank, failed_leader);
        fflush(stdout);

        // Start election: send own rank as initial token
        token = rank;
        int next = (rank + 1) % size;
        printf(">>> [Process %d] Sending ELECTION token (initial value: %d) to Process %d\n", rank, token, next);
        fflush(stdout);
        MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);

        // Wait for token to return from predecessor
        int prev = (rank - 1 + size) % size;
        MPI_Recv(&token, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf(">>> [Process %d] ELECTION token returned! Highest rank seen: %d\n", rank, token);
        fflush(stdout);

        // Declare winner
        elected_leader = token;
        printf(">>> [Process %d] ELECTION COMPLETE. New Coordinator is Process %d\n", rank, elected_leader);
	 fflush(stdout);

        // Broadcast new leader to all other processes
        int i;
        for (i = 0; i < size; i++) {
            if (i != rank) {
                printf(">>> [Process %d] Broadcasting new Coordinator (%d) to Process %d\n", rank, elected_leader, i);
                fflush(stdout);
                MPI_Send(&elected_leader, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            }
        }

    } else {
        // Non-initiator: receive, update, forward
        int prev = (rank - 1 + size) % size;
        int next = (rank + 1) % size;

        printf("[Process %d] Waiting to receive ELECTION token from Process %d...\n", rank, prev);
        fflush(stdout);

	 // If I am NOT the failed leader, consider my rank for leadership
        if (rank != failed_leader && rank > token) {
            printf("[Process %d] Updating token: My rank (%d) is higher than current max (%d)\n", rank, rank, token);
            token = rank;
        } else if (rank == failed_leader) {
            printf("[Process %d] I am the FAILED LEADER — not updating token.\n", rank);
        } else {
            printf("[Process %d] Not updating token (current max %d >= my rank %d)\n", rank, token, rank);
        }

        // Forward token to next process
        printf("[Process %d] Forwarding ELECTION token (%d) to Process %d\n", rank, token, next);
        fflush(stdout);
        MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
	
        // Receive final elected leader from initiator
        MPI_Recv(&elected_leader, 1, MPI_INT, initiator, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Process %d] Received broadcast: New Coordinator is Process %d\n", rank, elected_leader);
        fflush(stdout);
    }

    // All processes synchronize and print final result
    MPI_Barrier(MPI_COMM_WORLD);  // Optional: helps order output slightly
    sleep(1);                     // Optional: slows output for readability in small runs

    printf("\n=== FINAL RESULT ===\n");
    printf("Process %d: The elected Coordinator is Process %d\n", rank, elected_leader);
    printf("=====================\n\n");
    fflush(stdout);

    MPI_Finalize();
    return 0;
}