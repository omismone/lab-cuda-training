#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include<stdio.h>

#include "omislib.cuh"

__global__ void mulKernel(int* c, int* a, int* b, unsigned int  size)
{
    unsigned int i = threadIdx.x;
    unsigned int j = threadIdx.y;

    int tmp = 0;
    for (int k = 0; k < size; k++) {
        tmp += a[i * size + k] * b[k * size + j];
    }
    c[i * size + j] = tmp;
}

// Helper function for using CUDA to add arrtors in parallel.
void matrixMul(struct mat* c, struct mat* a, struct mat* b)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);

    // Allocate GPU buffers for three arrtors (two input, one output)    .
    cudaMalloc((void**)&dev_c, c->size[0] * c->size[1] * sizeof(int));

    cudaMalloc((void**)&dev_a, a->size[0] * a->size[1] * sizeof(int));

    cudaMalloc((void**)&dev_b, b->size[0] * b->size[1] * sizeof(int));

    // Copy input arrtors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a->val, a->size[0] * a->size[1] * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_b, b->val, b->size[0] * b->size[1] * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(c->size[0], c->size[1], 1);
    dim3 dimGrid(1, 1, 1);                                                     //to do: roba con più blocchi(tile). stesso concetto dello scorso es ma in 2 dimensioni.

    // Launch a kernel on the GPU with one thread for each element.
    mulKernel << <dimGrid, dimBlock >> > (dev_c, dev_a, dev_b, c->size[0]);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaDeviceSynchronize();

    // Copy output arrtor from GPU buffer to host memory.
    cudaMemcpy(c->val, dev_c, c->size[0] * c->size[1] * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}