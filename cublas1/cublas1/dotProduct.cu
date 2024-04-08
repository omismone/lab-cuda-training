#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <stdio.h>

#include "omislib.cuh"

void dotProduct(float* c, struct vec* a, struct vec* b)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;

    cublasHandle_t handle;

    cudaSetDevice(0);

    cublasCreate(&handle);
    
    cudaMalloc((void**)&dev_a, a->size * sizeof(float));
    cudaMalloc((void**)&dev_b, b->size * sizeof(float));
    cudaMalloc((void**)&dev_c, sizeof(float));

    cudaMemcpy(dev_a, a->val, a->size *  sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b->val, b->size * sizeof(float), cudaMemcpyHostToDevice);

    cublasStatus_t stat = cublasSdot(handle, a->size, dev_a, 1, dev_b, 1, dev_c);

    cudaMemcpy(c, dev_c, sizeof(float), cudaMemcpyDeviceToHost);
     
    cublasDestroy(handle);
    cudaFree(dev_a);
    cudaFree(dev_b);
}