## part2

**Q1:** produce a graph of speedup compared to the reference sequential implementation as a function of the number of threads used **FOR VIEW 1**.

**Answer:**
num of thread | speed up
:--------------:|:---------:
2|1.95
3|1.63
4|2.38

The graph of speedup
![](https://i.imgur.com/e7VIsyC.png)

From the data, the speedup and the number of threads used are not linearly related.
**reason:**

1. Different threads take on different amounts of work but the performance bottleneck is the time taken by the slowest thread to complete

**Q2:** How do your measurements explain the speedup graph you previously created?

**Answer:**
In the worst-performing three-threaded case, the execution time of the three threads is counted as follows:

```
thread 0 Time : 0.272314 s
thread 2 Time : 0.275829 s
thread 1 Time : 0.467044 s
thread 0 Time : 0.277368 s
thread 2 Time : 0.278432 s
thread 1 Time : 0.467802 s
thread 0 Time : 0.273191 s
thread 2 Time : 0.276238 s
thread 1 Time : 0.467676 s
thread 0 Time : 0.277246 s
thread 2 Time : 0.278408 s
thread 1 Time : 0.467839 s
thread 0 Time : 0.272846 s
thread 2 Time : 0.275855 s
thread 1 Time : 0.467017 s
```

Obviously thread 0 and thread 1 take up less work than thread 2, and this distribution causes thread 2 to become a performance bottleneck, so the overall performance is not greatly improved

**Q3:** describe your approach to parallelization and report the final 4-thread speedup obtained.

**Answer:**
The 4-thread speedup is **3.79x**
Because if each thread completes a part of the image, some threads are faster and some are slower due to the different computational complexity of different parts of the image, in order to distribute the work more fairly among the threads, each thread computes one row of the image at a time and skips the parts that have already been computed by other threads. This way the work is distributed more evenly.

**Q4:** Is performance noticeably greater than when running with four threads? Why or why not?

**Answer:**
4-thread speedup is 3.79
8-thread speedup is 3.64

**4-threaded performance is better than 8-threaded.**
**reason:**
Theoretically the performance is better in the 8-thread case than in the 4-thread case, but the reason for the fairness of the work distribution of the hesitant threads is that in the 8-thread case, there is more uneven distribution of the workload. While the work distribution of 4 threads is more even, so the performance of 8 threads is not as good as that of 4 threads.

The following statistics show that with 4 threads, the thread with the least work executes at 0.47x the thread with the most work at 0.5x. The distribution is fairly even. Under 8 threads, the least worked thread executes at 0.2x and the most worked thread executes at 0.4x.
4 threads per thread execution time:

```
thread 1 Time : 0.479094 s
thread 3 Time : 0.480254 s
thread 2 Time : 0.481100 s
thread 0 Time : 0.481566 s
thread 3 Time : 0.475835 s
thread 0 Time : 0.479678 s
thread 2 Time : 0.481450 s
thread 1 Time : 0.477735 s
thread 3 Time : 0.468441 s
thread 0 Time : 0.471635 s
thread 1 Time : 0.484926 s
thread 2 Time : 0.489638 s
thread 0 Time : 0.474288 s
thread 2 Time : 0.489496 s
thread 1 Time : 0.508218 s
thread 3 Time : 0.509383 s
thread 1 Time : 0.477347 s
thread 3 Time : 0.479201 s
thread 0 Time : 0.481433 s
thread 2 Time : 0.482653 s
```

8 threads per thread execution time:

```
thread 3 Time : 0.231065 s
thread 2 Time : 0.234402 s
thread 1 Time : 0.374372 s
thread 4 Time : 0.339079 s
thread 7 Time : 0.418359 s
thread 0 Time : 0.469135 s
thread 5 Time : 0.393547 s
thread 6 Time : 0.461809 s
thread 3 Time : 0.237107 s
thread 0 Time : 0.382432 s
thread 1 Time : 0.429030 s
thread 2 Time : 0.432676 s
thread 6 Time : 0.393143 s
thread 7 Time : 0.414942 s
thread 5 Time : 0.442873 s
thread 4 Time : 0.389285 s
thread 3 Time : 0.231211 s
thread 2 Time : 0.234515 s
thread 4 Time : 0.291167 s
thread 5 Time : 0.330279 s
thread 7 Time : 0.431791 s
thread 0 Time : 0.456583 s
thread 1 Time : 0.466220 s
thread 6 Time : 0.429704 s
thread 2 Time : 0.239466 s
thread 3 Time : 0.239902 s
thread 6 Time : 0.384666 s
thread 1 Time : 0.431126 s
thread 4 Time : 0.320200 s
thread 7 Time : 0.413417 s
thread 0 Time : 0.476952 s
thread 5 Time : 0.418549 s
thread 3 Time : 0.236747 s
thread 0 Time : 0.382205 s
thread 2 Time : 0.428794 s
thread 1 Time : 0.430142 s
thread 7 Time : 0.389549 s
thread 6 Time : 0.422687 s
thread 5 Time : 0.446985 s
thread 4 Time : 0.390024 s
```
