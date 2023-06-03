#include <fcntl.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
  // --- DON'T TOUCH ---
  MPI_Init(&argc, &argv);
  double start_time = MPI_Wtime();
  double pi_result;
  long long int tosses = atoi(argv[1]);
  int world_rank, world_size;
  long long int allNumber = 0;
  // ---

  // TODO: MPI init
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  long long int eachTosses = tosses / world_size;
  long long int reminder = tosses % world_size;

  int fd = open("/dev/urandom", O_RDONLY);
  unsigned int seed = 0;
  read(fd, (char *)&seed, sizeof(int));
  close(fd);
  long long int number = 0;
  for (int i = 0; i < eachTosses; i++) {
    double x = -1.0 + rand_r(&seed) / (double)(RAND_MAX / (1.0 - (-1.0)));
    double y = -1.0 + rand_r(&seed) / (double)(RAND_MAX / (1.0 - (-1.0)));
    double dinstant = x * x + y * y;
    if (dinstant <= 1) {
      number++;
    }
  }

  if (world_rank > 0) {
    // TODO: MPI workers
    MPI_Request req[world_size];
    MPI_Isend(&number, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD,
              &req[world_rank]);
  } else if (world_rank == 0) {
    // TODO: non-blocking MPI communication.
    // Use MPI_Irecv, MPI_Wait or MPI_Waitall.

    MPI_Request requests[world_size];
    MPI_Status status[world_size];
    int result[world_size];
    for (int i = 1; i < world_size; i++) {
      MPI_Irecv(&result[i], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD,
                &requests[i]);
    }
    allNumber += number;
    MPI_Waitall(world_size - 1, requests + 1, status + 1);
    for (int i = 1; i < world_size; i++) {
      allNumber += result[i];
    }
  }

  if (world_rank == 0) {
    // TODO: PI result
    pi_result = 4 * allNumber / (double)tosses;

    // --- DON'T TOUCH ---
    double end_time = MPI_Wtime();
    printf("%lf\n", pi_result);
    printf("MPI running time: %lf Seconds\n", end_time - start_time);
    // ---
  }

  MPI_Finalize();
  return 0;
}
