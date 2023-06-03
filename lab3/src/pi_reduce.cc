#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---
    long long int eachTosses = tosses / world_size;
    long long int reminder = tosses % world_size;
    long long int allNumber = 0;

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    eachTosses = (world_rank == 0) ? (eachTosses + reminder) : eachTosses;
    // TODO: use MPI_Reduce
    int fd = open("/dev/urandom", O_RDONLY);
    unsigned int seed = 0;
    seed *= world_rank;
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
    MPI_Reduce(&number, &allNumber, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  
    if (world_rank == 0)
    {
        // TODO: PI result
       pi_result = 4 * (allNumber) / (double) tosses; 

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
