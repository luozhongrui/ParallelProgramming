#include <fcntl.h>
#include <iostream>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
long long int numCircle = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void *esitmeatePi(void *each) {
  int fd = open("/dev/urandom", O_RDONLY);
  unsigned int seed = 0;
  read(fd, (char *)&seed, sizeof(int));
  close(fd);

  long long int iter = *(long long int *)each;
  long long int number = 0;
  for (int i = 0; i < iter; i++) {
    double x = -1.0 + rand_r(&seed) / (double)(RAND_MAX / (1.0 - (-1.0)));
    double y = -1.0 + rand_r(&seed) / (double)(RAND_MAX / (1.0 - (-1.0)));
    double dinstant = x * x + y * y;
    if (dinstant <= 1) {
      number++;
    }
  }
  pthread_mutex_lock(&lock);
  numCircle += number;
  pthread_mutex_unlock(&lock);
  pthread_exit(0);
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "[Usage:] ./pi.out [num of thread] [tosses]" << std::endl;
    return 0;
  }

  int numTread = atoi(argv[1]);
  long long int tosses = atoll(argv[2]);
  long long int eachtosses = tosses / numTread;
  long long int remind = tosses % numTread;

  pthread_t thread[numTread];
  for (int i = 0; i < numTread; i++) {
    if (i == numTread - 1) {
      eachtosses = eachtosses + remind;
    }
    pthread_create(&thread[i], NULL, esitmeatePi, &eachtosses);
  }
  for (int i = 0; i < numTread; i++) {
    pthread_join(thread[i], NULL);
  }
  double pi = 4 * numCircle / (double)tosses;
  printf("%.6f\n", pi);

  return 0;
}