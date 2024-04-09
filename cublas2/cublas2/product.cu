#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#include "omislib.cuh"



void productMatVec(struct vec* c, struct mat* a, struct vec* b)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;

    cublasHandle_t handle;

    cudaSetDevice(0);

    cublasCreate(&handle);

    cudaMalloc((void**)&dev_a, a->size[0] * a->size[1] * sizeof(float));
    cudaMalloc((void**)&dev_b, b->size * sizeof(float));
    cudaMalloc((void**)&dev_c, c->size * sizeof(float));

    cudaMemcpy(dev_a, a->val, a->size[0] * a->size[1] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b->val, b->size * sizeof(float), cudaMemcpyHostToDevice);
    
    const float alpha = 1.0;
    const float beta = 0.0;
    cublasStatus_t stat = cublasSgemv(handle, CUBLAS_OP_T, a->size[1], a->size[0] , &alpha, dev_a, a->size[1], dev_b, 1, &beta, dev_c, 1);

    cudaMemcpy(c->val, dev_c, c->size * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

void productMatMat(struct mat* c, struct mat* a, struct mat* b)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;

    cublasHandle_t handle;
    
    cudaSetDevice(0);

    cublasCreate(&handle);

    cudaMalloc((void**)&dev_a, a->size[0] * a->size[1] * sizeof(float));
    cudaMalloc((void**)&dev_b, b->size[0] * b->size[1] * sizeof(float));
    cudaMalloc((void**)&dev_c, c->size[0] * c->size[1] * sizeof(float));

    cudaMemcpy(dev_a, a->val, a->size[0] * a->size[1] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b->val, b->size[0] * b->size[1] * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0;
    const float beta = 0.0;
    int m = b->size[1];
    int n = a->size[0];
    int k = b->size[0];
    cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dev_b, n, dev_a, k, &beta, dev_c, n);

    cudaMemcpy(c->val, dev_c, c->size[0] * c->size[1] * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(dev_a);
    cudaFree(dev_b);    
    cudaFree(dev_c);
}