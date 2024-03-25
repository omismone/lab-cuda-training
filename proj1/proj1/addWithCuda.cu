#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include<stdio.h>

#include "omislib.cuh"

#define THREADPERBLOCK 5



__global__ void addKernel(int* c, int* a, int* b, unsigned int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size)    
        c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add arrtors in parallel.
void addWithCuda(struct arr* c, struct arr* a, struct arr* b)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);

    // Allocate GPU buffers for three arrtors (two input, one output)    .
    cudaMalloc((void**)&dev_c, c->size * sizeof(int));

    cudaMalloc((void**)&dev_a, a->size * sizeof(int));

    cudaMalloc((void**)&dev_b, b->size * sizeof(int));

    // Copy input arrtors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a->val, a->size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_b, b->val, b->size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(THREADPERBLOCK, 1, 1);
    int num = c->size / THREADPERBLOCK;
    if (c->size % THREADPERBLOCK != 0)
        num += 1;
    dim3 dimGrid(num, 1, 1);

    // Launch a kernel on the GPU with one thread for each element.
    addKernel <<<dimGrid, dimBlock >> > (dev_c, dev_a, dev_b, c->size);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaDeviceSynchronize();

    // Copy output arrtor from GPU buffer to host memory.
    cudaMemcpy(c->val, dev_c, c->size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}