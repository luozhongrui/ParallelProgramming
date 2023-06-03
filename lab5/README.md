# OpenCL Programming

**Q1:** Explain your implementation. How do you optimize the performance of convolution?
**Answer:**

Assign jobs according to serial version and let each thread calculate one pixel value. Consider data reuse, and memory joint access to swap index order within the loop.The filter data size is not large, so the filter data is placed in constant memory to improve the read speed.
