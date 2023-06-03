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

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int eachTosses = tosses / world_size;
    long long int reminder = tosses % world_size;
    if(world_rank == 0) eachTosses += reminder;
    // TODO: binary tree redunction
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
   int remain = world_size;
   int  half;
   int  temp;
   long long int  sum = number;
   while(remain != 1){
     half = remain / 2;
    if(world_rank < half) {
     MPI_Recv(&temp, 1, MPI_LONG_LONG, world_rank + half, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
     sum += temp;
    }
   else{
   MPI_Send(&sum, 1, MPI_LONG_LONG, world_rank - half, 0, MPI_COMM_WORLD);
   break;
   }
  remain = half;
}
   
    

    if (world_rank == 0)
    {
        // TODO: PI result
       pi_result = 4 * sum / (double) tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
