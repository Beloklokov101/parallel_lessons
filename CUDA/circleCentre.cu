#include <cuda.h>
#include <stdio.h>

#define CUDA_FLOAT float
// #define BLOCK_SIZE 128 // % 32 == 0
#define BLOCK_SIZE 64 // % 32 == 0
#define THREAD_SIZE 32
#define GRID_SIZE 10

__global__ void pi_kern(CUDA_FLOAT *resUp, CUDA_FLOAT *resDown)
{
    int n = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    CUDA_FLOAT x0 = n * 1.f / (BLOCK_SIZE * GRID_SIZE); // Начало отрезка интегрирования
    CUDA_FLOAT y0 = sqrtf(1 - x0 * x0);
    CUDA_FLOAT dx = 1.f / (1.f * BLOCK_SIZE * GRID_SIZE * THREAD_SIZE); // Шаг интегрирования
    CUDA_FLOAT sUp = 0, sDown = 0; // Значение интеграла по отрезку, данному текущему треду
    CUDA_FLOAT x1, y1;
    for (int i=0; i < THREAD_SIZE; ++i)
    {
        x1 = x0 + dx;
        y1 = sqrtf(1 - x1 * x1);
        sUp += (y0 * x0 + y1 * x1) * dx / 2.f;
        sDown += (y0 + y1) * dx / 2.f; // Площадь трапеции
        x0 = x1; y0 = y1;
    }
    printf("%f, %f\n", sUp, sDown);
    resUp[n] = sUp; // Запись результата в глобальную память
    resDown[n] = sDown; // Запись результата в глобальную память
}

int main()
{
    CUDA_FLOAT * devPtrUp; // pointer device memory
    CUDA_FLOAT * hostPtrUp;
    CUDA_FLOAT * devPtrDown; // pointer device memory
    CUDA_FLOAT * hostPtrDown;
    // allocate device memory
    cudaMalloc ( (void **) &devPtrUp, BLOCK_SIZE * GRID_SIZE * sizeof ( CUDA_FLOAT ) );
    hostPtrUp = (CUDA_FLOAT *)malloc(BLOCK_SIZE * GRID_SIZE * sizeof(CUDA_FLOAT));
    cudaMalloc ( (void **) &devPtrDown, BLOCK_SIZE * GRID_SIZE * sizeof ( CUDA_FLOAT ) );
    hostPtrDown = (CUDA_FLOAT *)malloc(BLOCK_SIZE * GRID_SIZE * sizeof(CUDA_FLOAT));

    pi_kern<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>> (devPtrUp, devPtrDown);
    // copy results from device to host memory
    cudaMemcpy ( hostPtrUp, devPtrUp, BLOCK_SIZE * GRID_SIZE * sizeof(CUDA_FLOAT), cudaMemcpyDeviceToHost );
    cudaMemcpy ( hostPtrDown, devPtrDown, BLOCK_SIZE * GRID_SIZE * sizeof(CUDA_FLOAT), cudaMemcpyDeviceToHost );
    
    CUDA_FLOAT sumUp = 0, sumDown = 0;
    for (int i = 0; i < BLOCK_SIZE * GRID_SIZE; i++)
    {
        sumUp += hostPtrUp[i];
        sumDown += hostPtrDown[i];
    }
    printf("%f\n", sumUp / sumDown);
    // Answer is \frac{4}{3\pi}

    // process data
    // dim3 block = dim3(BLOCK_SIZE);
    // dim3 grid = dim3(N / BLOCK_SIZE);

    // copy results from device to host
    // cudaMemcpy ( hostPtr, devPtr, 256*sizeof( float ), cudaMemcpyDeviceToHost );
    // free device memory
    cudaFree ( devPtrUp );
    free(hostPtrUp);
    cudaFree ( devPtrDown );
    free(hostPtrDown);

    return 1;
}