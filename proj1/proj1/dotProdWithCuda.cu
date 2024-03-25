#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include<stdio.h>

#include "omislib.cuh"

#define THREADPERBLOCK 5



__global__ void prodKernel(int* c, int* a, int* b, unsigned int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] * b[i];
}

/*
*   sum each element of an array and puts the result in the first element of the array itself
*/
__global__ void sumKernel(int *arr, unsigned int size)
{
    for (int i = 1; i < size; i++) {
        arr[0] += arr[i];
    }
}

// Helper function for using CUDA to add arrtors in parallel.
int dotProdWithCuda(struct arr* a, struct arr* b)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_tmp = 0;
    int sum;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);

    // Allocate GPU buffers for three arrtors (two input, one output)    .
    cudaMalloc((void**)&dev_tmp, a->size * sizeof(int));

    cudaMalloc((void**)&dev_a, a->size * sizeof(int));

    cudaMalloc((void**)&dev_b, b->size * sizeof(int));

    // Copy input arrtors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a->val, a->size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b->val, b->size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(THREADPERBLOCK, 1, 1);
    int num = a->size / THREADPERBLOCK;
    if (a->size % THREADPERBLOCK != 0)
        num += 1;
    dim3 dimGrid(num, 1, 1);

    // Launch a kernel on the GPU with one thread for each element.
    prodKernel << <dimGrid, dimBlock >> > (dev_tmp, dev_a, dev_b, a->size);

    // Launch a kernel on the GPU that sum each element.
    sumKernel << <1, 1 >> > (dev_tmp, a->size);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaDeviceSynchronize();

    // Copy output from GPU buffer to host memory.
    cudaMemcpy(&sum, &dev_tmp[0], sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_tmp);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return sum;
}