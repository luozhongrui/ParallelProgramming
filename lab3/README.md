# MPI Programming

## part1

**Q1:**

1. How do you control the number of MPI processes on each node?
   **Answer:**
   Use the `slots` parameter in the hostfile to control how many processes are allowed to execute in each node.

2. Which functions do you use for retrieving the rank of an MPI process and the total number of processes?
   **Answer:**
   the function which retrive the rank of an MPI process is `MPI_Comm_rank(MPI_COMM_WORLD, &world_rank)`
   the function which retrive the total number of processes is `MPI_Comm_size(MPI_COMM_WORLD, &world_size)`

**Q2:**

1. Why `MPI_Send` and `MPI_Recv` are called “blocking” communication?
   **Answer:**
   Because `MPI_Send` and `MPI_Recv` functions need to wait for all data to be copied to the buffer before returning.
2. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.
   ![](https://i.imgur.com/lTjDyBc.png)

**Q3:**

1. Measure the performance (execution time) of the code for 2, 4, 8, 16 MPI processes and plot it.
   ![](https://i.imgur.com/f1K0vwM.png)

2. How does the performance of binary tree reduction compare to the performance of linear reduction?
   **Answer**
   ![](https://i.imgur.com/kTDeJhl.png)
   Since the number of communications for the tree reduction method is less than that for the linear reduction method. This trend becomes more evident as the number of processes increases.

3. Increasing the number of processes, which approach (linear/tree) is going to perform better? Why? Think about the number of messages and their costs.
   **Answer**
   **Tree reduction method is better.**
   Because the number of communications for the tree reduction method is halved each time, and the number of communications for the linear reduction method is the number of processes minus one, the tree reduction method performs better as the number of processes increases.

**Q4:**

1. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.
   ![](https://i.imgur.com/B08DJar.png)

2. What are the MPI functions for non-blocking communication?
   **Answer**

```
MPI_Isend
MPI_Irecv
MPI_Wait
MPI_Waitall
```

3. How the performance of non-blocking communication compares to the performance of blocking communication?
   ![](https://i.imgur.com/kMPsVqt.png)

**Q5:**

1. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.
   ![](https://i.imgur.com/eBVCp6K.png)

**Q6:**

1. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.
   ![](https://i.imgur.com/Ogj0WLn.png)

## part2

**Q7**

1. Describe what approach(es) were used in your MPI matrix multiplication for each data set.
   **Answer**

- In the first dataset because the matrix size is relatively small, so MPI communication cost will be greater than the cost of calculation, so choose to calculate on a node not to use MPI distribution to calculate in different nodes, and use matrix transposition in the calculation process to improve data locality to reduce cache miss, the loop part of the calculation process using the size of 16 loop unrolling.
- For the second dataset, the matrix size is larger, so we evenly distribute each node to calculate the value of the matrix part and finally pass it back to process 0 for output. At each compute node, the matrix transpose and loop unrolling methods consistent with case 1 are used to accelerate.
