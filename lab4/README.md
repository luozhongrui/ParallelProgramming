# CUDA Programming

**Q1**
What are the pros and cons of the three methods? Give an assumption about their performances.

**Answer**

- **Assumption:** The second method performs best, the first method second, and the third method slowest.
- **method1:**
  - _pros:_
    1. Each thread handles one pixel idea intuition easy to implement.
    2. Evenly distribute the workload of the thread when the image size is smaller than the number of devices SM.
    3. Good performance when the image size is small.
  - _cons:_
    1. The `malloc` allocated memory needs to be copied to pinned memory and then to device memory when copying to device memory.
    2. The memory allocated by `cudaMalloc` can only be described in one dimension.
- **method 2:**
  - _pros:_
    1. The memory allocated by `cudaHostAlloc` is pinned memory, which can be quickly copied to the device side to reduce memory access latency.
    2. The memory allocated by cudaMallocPitch can be described in a two-dimensional way to facilitate the storage of image data.
  - _cons:_
    1. Pinned memory is resident in memory and requires a large enough memory space on the system to support it.
    2. When memory is limited on the deivce side, blank conflicts can easily occur, thus affecting performance.
- **method 3:**
  - _pros:_
    1. Each thread handles one set of pixels, allowing the device to process larger images simultaneously.
  - _cons:_
    1.  Due to the different complexity of each group of pixels, it may lead to uneven work distribution with higher load on some threads.

**Q2**
How are the performances of the three methods? Plot a chart to show the differences among the three methods

- for VIEW 1
  ![](https://i.imgur.com/iOOADsb.png)

- for VIEW 2
  ![](https://i.imgur.com/AbhT2hG.png)

- for different maxIteration (1000, 10000, and 100000).

  - for 1000 maxIteration

    - ![](https://i.imgur.com/Y1WrKZc.png)

  - for 10000 maxIteration

    - ![](https://i.imgur.com/Ry9ckYS.png)

  - for 100000 maxIteration
    - ![](https://i.imgur.com/CwSGRn8.png)

**Q3**
Explain the performance differences thoroughly based on your experimental results. Does the results match your assumption? Why or why not.
**Answer**

1. As assumed the third method is the slowest because the work is most unevenly distributed.
2. In fact, the first method performs better than the second one, mainly because the image size is not particularly large so the overhead of `cudaHostAlloc` is larger than the overhead of the copied data itself.

**Q4**
Can we do even better? Think a better approach and explain it.
**Answer**
Dip all hardware resources as much as possible so that all blocks inside the grid are involved in the computation. Therefore, the number of threads per block is reduced and each thread knows to compute one pixel. Also consider whether the overhead of thread allocation and scheduling will outweigh the benefit of parallel computing itself
