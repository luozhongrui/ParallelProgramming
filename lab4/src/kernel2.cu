#include <cstddef>
#include <cuda.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int resX, int maxIterations, int *output) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int tx = (blockIdx.x * blockDim.x + threadIdx.x); 
    int ty = (blockIdx.y * blockDim.y + threadIdx.y);
    float x = lowerX + tx * stepX;
    float y = lowerY + ty * stepY;
    
    float z_re = x, z_im = y;
    int i;
    for(i = 0; i < maxIterations; ++i){
        if(z_re * z_re + z_im * z_im > 4.f)
            break;
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
    }
    int *row = (int*)((char*)output + ty * resX);
    row[tx] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int dataSize = resX * resY * sizeof(int);
    int *hostData = NULL;
    int *deviceData = NULL;
    size_t pitch;
    cudaHostAlloc(&hostData, dataSize, cudaHostAllocMapped);
    cudaMallocPitch(&deviceData, &pitch, resX * sizeof(int), resY);
    dim3 threadPerBlock(32, 25);
    dim3 blockPerGrid(resX / threadPerBlock.x, resY / threadPerBlock.y);
   mandelKernel<<<blockPerGrid, threadPerBlock>>>(lowerX,  lowerY,  stepX,  stepY,  pitch, maxIterations, deviceData);
   cudaMemcpy2D(hostData, resX * sizeof(int), deviceData, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
   memcpy(img, hostData, dataSize);
   cudaFree(deviceData);
   cudaFreeHost(hostData);
}
