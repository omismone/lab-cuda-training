#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include<stdio.h>

#include "omislib.cuh"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
/*
*   Device function that elaborate the result of matrix mul in position (x,y).
*/
__global__ void mulKernel(int* c, int* a, int* b, const unsigned int block_size, unsigned int  tile_w, unsigned int tile_h)
{
    unsigned int i = blockIdx.x * tile_w + threadIdx.x;
    unsigned int j = blockIdx.y * tile_h + threadIdx.y;

    int tmp = 0;
    for (int k = 0; k < block_size; k++) {
        tmp += a[i * block_size + k] * b[k * block_size + j];
    }
    c[i * block_size + j] = tmp;
}

// Helper function for using CUDA to add arrtors in parallel.
void matrixMul(struct mat* c, struct mat* a, struct mat* b)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);

    // Allocate GPU buffers for three arrtors (two input, one output).
    cudaMalloc((void**)&dev_c, c->size[0] * c->size[1] * sizeof(int));

    cudaMalloc((void**)&dev_a, a->size[0] * a->size[1] * sizeof(int));

    cudaMalloc((void**)&dev_b, b->size[0] * b->size[1] * sizeof(int));

    // Copy input arrtors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a->val, a->size[0] * a->size[1] * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_b, b->val, b->size[0] * b->size[1] * sizeof(int), cudaMemcpyHostToDevice);

    const unsigned int BLOCK_SIZE = MIN(a->size[0], 32);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);   //optimal size
    int tile_w = c->size[0] / BLOCK_SIZE;
    if (c->size[0] % BLOCK_SIZE != 0)
        tile_w += 1;
    int tile_h = c->size[1] / BLOCK_SIZE;
    if (c->size[1] % BLOCK_SIZE != 0)
        tile_h += 1;

    dim3 dimGrid(tile_w, tile_h, 1);

    // Launch a kernel on the GPU with one BLOCK for each element.
    mulKernel << <dimGrid, dimBlock >> > (dev_c, dev_a, dev_b, BLOCK_SIZE, tile_w, tile_h);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaDeviceSynchronize();

    // Copy output arrtor from GPU buffer to host memory.
    cudaMemcpy(c->val, dev_c, c->size[0] * c->size[1] * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}