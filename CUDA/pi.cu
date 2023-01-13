#include <cuda.h>
#include <stdio.h>

#define CUDA_FLOAT float
// #define BLOCK_SIZE 128 // % 32 == 0
#define BLOCK_SIZE 64 // % 32 == 0
#define THREAD_SIZE 32
#define GRID_SIZE 10
__global__ void pi_kern(CUDA_FLOAT *res)
{
    int n = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    CUDA_FLOAT x0 = n * 1.f / (BLOCK_SIZE * GRID_SIZE); // Начало отрезка интегрирования
    CUDA_FLOAT y0 = sqrtf(1 - x0 * x0);
    CUDA_FLOAT dx = 1.f / (1.f * BLOCK_SIZE * GRID_SIZE * THREAD_SIZE); // Шаг интегрирования
    CUDA_FLOAT s = 0; // Значение интеграла по отрезку, данному текущему треду
    CUDA_FLOAT x1, y1;
    for (int i=0; i < THREAD_SIZE; ++i)
    {
        x1 = x0 + dx;
        y1 = sqrtf(1 - x1 * x1);
        s += (y0 + y1) * dx / 2.f; // Площадь трапеции
        x0 = x1; y0 = y1;
    }
    printf("%f \n", s);
    res[n] = s; // Запись результата в глобальную память
}

int main()
{
    CUDA_FLOAT * devPtr; // pointer device memory
    CUDA_FLOAT * hostPtr;
    // allocate device memory
    cudaMalloc ( (void **) &devPtr, BLOCK_SIZE * GRID_SIZE * sizeof ( CUDA_FLOAT ) );
    hostPtr = (CUDA_FLOAT *)malloc(BLOCK_SIZE * GRID_SIZE * sizeof(CUDA_FLOAT));

    pi_kern<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>> (devPtr);
    // copy results from device to host memory
    cudaMemcpy ( hostPtr, devPtr, BLOCK_SIZE * GRID_SIZE * sizeof(CUDA_FLOAT), cudaMemcpyDeviceToHost );
    
    CUDA_FLOAT sum = 0;
    for (int i = 0; i < BLOCK_SIZE * GRID_SIZE; i++)
    {
        sum += hostPtr[i];
    }
    printf("%f\n", 4 * sum);

    // process data
    // dim3 block = dim3(BLOCK_SIZE);
    // dim3 grid = dim3(N / BLOCK_SIZE);

    // copy results from device to host
    // cudaMemcpy ( hostPtr, devPtr, 256*sizeof( float ), cudaMemcpyDeviceToHost );
    // free device memory
    cudaFree ( devPtr );
    free(hostPtr);

    return 1;
}