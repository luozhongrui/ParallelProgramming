#include <mpi.h>
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr,
                        int **b_mat_ptr) {
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  if (my_rank == 0) {
    scanf("%d%d%d", n_ptr, m_ptr, l_ptr);
    int n = *n_ptr, m = *m_ptr, l = *l_ptr;
    *a_mat_ptr = (int *)malloc(n * m * sizeof(int));
    *b_mat_ptr = (int *)malloc(m * l * sizeof(int));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        scanf("%d", (*a_mat_ptr) + i * m + j);
      }
    }
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < l; j++) {
        scanf("%d", (*b_mat_ptr) + i * l + j);
      }
    }
  }
  MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void destruct_matrices(int *a_mat, int *b_mat) {
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  if (my_rank == 0) {
    free(a_mat);
    free(b_mat);
  }
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat,
                     const int *b_mat) {
  int my_rank;
  int world_size;
  int *b, *c, *buffer, *ans;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int line = n / world_size;
  // a = (int *)malloc(sizeof(int) * m * line);
  b = (int *)malloc(sizeof(int) * m * l);
  c = (int *)malloc(sizeof(int) * n * l);
  buffer = (int *)malloc(sizeof(int) * line * m);
  ans = (int *)malloc(sizeof(int) * line * l);
  memset(c, 0, sizeof(int) * n * l);
  memset(ans, 0, sizeof(int) * line * l);
  if (n <= 500 || m <= 500 || l <= 500) {
    if (my_rank == 0) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
          int k;
          for (k = 0; k + 15 < l; k += 16) {
            c[i * l + k] += a_mat[i * m + j] * b_mat[j * l + k];
            c[i * l + k + 1] += a_mat[i * m + j] * b_mat[j * l + k + 1];
            c[i * l + k + 2] += a_mat[i * m + j] * b_mat[j * l + k + 2];
            c[i * l + k + 3] += a_mat[i * m + j] * b_mat[j * l + k + 3];
            c[i * l + k + 4] += a_mat[i * m + j] * b_mat[j * l + k + 4];
            c[i * l + k + 5] += a_mat[i * m + j] * b_mat[j * l + k + 5];
            c[i * l + k + 6] += a_mat[i * m + j] * b_mat[j * l + k + 6];
            c[i * l + k + 7] += a_mat[i * m + j] * b_mat[j * l + k + 7];
            c[i * l + k + 8] += a_mat[i * m + j] * b_mat[j * l + k + 8];
            c[i * l + k + 9] += a_mat[i * m + j] * b_mat[j * l + k + 9];
            c[i * l + k + 10] += a_mat[i * m + j] * b_mat[j * l + k + 10];
            c[i * l + k + 11] += a_mat[i * m + j] * b_mat[j * l + k + 11];
            c[i * l + k + 12] += a_mat[i * m + j] * b_mat[j * l + k + 12];
            c[i * l + k + 13] += a_mat[i * m + j] * b_mat[j * l + k + 13];
            c[i * l + k + 14] += a_mat[i * m + j] * b_mat[j * l + k + 14];
            c[i * l + k + 15] += a_mat[i * m + j] * b_mat[j * l + k + 15];
          }
          for (; k < l; k++) {
            c[i * l + k] += a_mat[i * m + j] * b_mat[j * l + k];
          }
        }
      }
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < l; j++) {
          printf("%d ", c[i * l + j]);
        }
        printf("\n");
      }
    }

  } else {
    if (my_rank == 0) {
      MPI_Request requests[world_size], recv[world_size];
      MPI_Status status[world_size];
      for (int i = 1; i < world_size; i++) {
        MPI_Isend(b_mat, m * l, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i]);
        MPI_Isend(a_mat + (i - 1) * line * m, m * line, MPI_INT, i, 1,
                  MPI_COMM_WORLD, &requests[i]);
      }

      for (int i = (world_size - 1) * line; i < n; i++) {
        for (int j = 0; j < m; j++) {
          int k;
          for (k = 0; k + 15 < l; k += 16) {
            c[i * l + k] += a_mat[i * m + j] * b_mat[j * l + k];
            c[i * l + k + 1] += a_mat[i * m + j] * b_mat[j * l + k + 1];
            c[i * l + k + 2] += a_mat[i * m + j] * b_mat[j * l + k + 2];
            c[i * l + k + 3] += a_mat[i * m + j] * b_mat[j * l + k + 3];
            c[i * l + k + 4] += a_mat[i * m + j] * b_mat[j * l + k + 4];
            c[i * l + k + 5] += a_mat[i * m + j] * b_mat[j * l + k + 5];
            c[i * l + k + 6] += a_mat[i * m + j] * b_mat[j * l + k + 6];
            c[i * l + k + 7] += a_mat[i * m + j] * b_mat[j * l + k + 7];
            c[i * l + k + 8] += a_mat[i * m + j] * b_mat[j * l + k + 8];
            c[i * l + k + 9] += a_mat[i * m + j] * b_mat[j * l + k + 9];
            c[i * l + k + 10] += a_mat[i * m + j] * b_mat[j * l + k + 10];
            c[i * l + k + 11] += a_mat[i * m + j] * b_mat[j * l + k + 11];
            c[i * l + k + 12] += a_mat[i * m + j] * b_mat[j * l + k + 12];
            c[i * l + k + 13] += a_mat[i * m + j] * b_mat[j * l + k + 13];
            c[i * l + k + 14] += a_mat[i * m + j] * b_mat[j * l + k + 14];
            c[i * l + k + 15] += a_mat[i * m + j] * b_mat[j * l + k + 15];
          }
          for (; k < l; ++k) {
            c[i * l + k] += a_mat[i * m + j] * b_mat[j * l + k];
          }
        }
      }

      for (int k = 1; k < world_size; k++) {
        MPI_Recv(ans, line * l, MPI_INT, k, 3, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        for (int i = 0; i < line; i++) {
          int j;
          for (j = 0; j + 15 < l; j += 16) {
            c[((k - 1) * line + i) * l + j] = ans[i * l + j];
            c[((k - 1) * line + i) * l + j + 1] = ans[i * l + j + 1];
            c[((k - 1) * line + i) * l + j + 2] = ans[i * l + j + 2];
            c[((k - 1) * line + i) * l + j + 3] = ans[i * l + j + 3];
            c[((k - 1) * line + i) * l + j + 4] = ans[i * l + j + 4];
            c[((k - 1) * line + i) * l + j + 5] = ans[i * l + j + 5];
            c[((k - 1) * line + i) * l + j + 6] = ans[i * l + j + 6];
            c[((k - 1) * line + i) * l + j + 7] = ans[i * l + j + 7];
            c[((k - 1) * line + i) * l + j + 8] = ans[i * l + j + 8];
            c[((k - 1) * line + i) * l + j + 9] = ans[i * l + j + 9];
            c[((k - 1) * line + i) * l + j + 10] = ans[i * l + j + 10];
            c[((k - 1) * line + i) * l + j + 11] = ans[i * l + j + 11];
            c[((k - 1) * line + i) * l + j + 12] = ans[i * l + j + 12];
            c[((k - 1) * line + i) * l + j + 13] = ans[i * l + j + 13];
            c[((k - 1) * line + i) * l + j + 14] = ans[i * l + j + 14];
            c[((k - 1) * line + i) * l + j + 15] = ans[i * l + j + 15];
          }
          for (; j < l; j++) {
            c[((k - 1) * line + i) * l + j] = ans[i * l + j];
          }
        }
      }

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < l; j++) {
          printf("%d ", c[i * l + j]);
        }
        printf("\n");
      }

    } else {
      MPI_Request requestb[world_size], requestbuff[world_size],
          send[world_size];
      MPI_Status statusb[world_size], statusbuff[world_size];
      MPI_Irecv(b, m * l, MPI_INT, 0, 0, MPI_COMM_WORLD, &requestb[my_rank]);
      MPI_Irecv(buffer, m * line, MPI_INT, 0, 1, MPI_COMM_WORLD,
                &requestbuff[my_rank]);
      MPI_Wait(&requestb[my_rank], &statusb[my_rank]);
      MPI_Wait(&requestbuff[my_rank], &statusbuff[my_rank]);

      for (int i = 0; i < line; i++) {
        for (int j = 0; j < m; j++) {
          int k;
          for (k = 0; k + 15 < l; k += 16) {
            ans[i * l + k] += buffer[i * m + j] * b[j * l + k];
            ans[i * l + k + 1] += buffer[i * m + j] * b[j * l + k + 1];
            ans[i * l + k + 2] += buffer[i * m + j] * b[j * l + k + 2];
            ans[i * l + k + 3] += buffer[i * m + j] * b[j * l + k + 3];
            ans[i * l + k + 4] += buffer[i * m + j] * b[j * l + k + 4];
            ans[i * l + k + 5] += buffer[i * m + j] * b[j * l + k + 5];
            ans[i * l + k + 6] += buffer[i * m + j] * b[j * l + k + 6];
            ans[i * l + k + 7] += buffer[i * m + j] * b[j * l + k + 7];
            ans[i * l + k + 8] += buffer[i * m + j] * b[j * l + k + 8];
            ans[i * l + k + 9] += buffer[i * m + j] * b[j * l + k + 9];
            ans[i * l + k + 10] += buffer[i * m + j] * b[j * l + k + 10];
            ans[i * l + k + 11] += buffer[i * m + j] * b[j * l + k + 11];
            ans[i * l + k + 12] += buffer[i * m + j] * b[j * l + k + 12];
            ans[i * l + k + 13] += buffer[i * m + j] * b[j * l + k + 13];
            ans[i * l + k + 14] += buffer[i * m + j] * b[j * l + k + 14];
            ans[i * l + k + 15] += buffer[i * m + j] * b[j * l + k + 15];
          }
          for (; k < l; k++) {
            ans[i * l + k] += buffer[i * m + j] * b[j * l + k];
          }
        }
      }
      MPI_Send(ans, line * l, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }
  }
}
