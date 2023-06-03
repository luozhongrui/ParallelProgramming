#include <CL/cl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageWidth * imageHeight * sizeof(float);


    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &status);

    cl_mem inputMemObj = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSize, NULL, &status);
    cl_mem outputMemObj = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, NULL, &status);
    cl_mem filterMemObj = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, &status);
    
    status = clEnqueueWriteBuffer(command_queue, inputMemObj, CL_TRUE, 0, imageSize, (void*)inputImage, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, filterMemObj, CL_TRUE, 0, filterSize, (void*)filter, 0, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(*program, "convolutiion", status);
    
    status = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&filterWidth);
    status = clSetKernelArg(kernel, 1, sizeof(filterMemObj), (void *)&filterMemObj);
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&imageHeight);
    status = clSetKernelArg(kernel, 3, sizeof(cl_int),(void *)&imageWidth);
    status = clSetKernelArg(kernel, 4, sizeof(inputMemObj), (void *)&inputMemObj);
    status = clSetKernelArg(kernel, 5, sizeof(outputMemObj), (void *)&outputMemObj);


    size_t local_item_size[2] = {25, 25};
    size_t global_item_size[2] = {imageWidth, imageHeight};

    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, 0, global_item_size, local_item_size, 0, NULL, NULL);

    status = clEnqueueReadBuffer(command_queue, outputMemObj, CL_TRUE, 0, imageSize,(void*)outputImage, NULL, NULL, NULL);

    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseMemObject(filterMemObj);
    clReleaseMemObject(inputMemObj);
    clReleaseMemObject(outputMemObj);
    
}
