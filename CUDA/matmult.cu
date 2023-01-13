#include <cuda.h>
#include <stdio.h>

#define CUDA_FLOAT float
// #define BLOCK_SIZE 128 // % 32 == 0
// #define BLOCK_SIZE 64 // % 32 == 0
// #define THREAD_SIZE 32
// #define GRID_SIZE 10

#define BS 32
#define WA 100
#define WB 100
#define HA 100
#define HB WA
#define WC WB
#define HC HA

__global__ void matmult(CUDA_FLOAT *matA, CUDA_FLOAT *matB, CUDA_FLOAT *matC)
{
    int BX = blockIdx.x;
    int BY = blockIdx.y;
    int TX = threadIdx.x;
    int TY = threadIdx.y;
    // printf("%d, %d, %d, %d\n", BX, BY, TX, TY);

    int Astart = BS * WA * BS * BY + BS * WA * TY;
    int Astep = 1;
    int Astop = Astart + BS * WA;

    int Bstart = BS * BX + TX;
    int Bstep = WB;

    CUDA_FLOAT res = 0;
    int i, j = Bstart;
    for (i = Astart; i < Astop; i += Astep)
    {
        // printf("%f, %f\n", matA[i], matB[j]);
        res += matA[i] * matB[j];
        j += Bstep;
    }
    // matC[BY * BS ] = res;
    // matC[WA * BS * BY + BS * BX + TX] = res;
    // printf("%f\n", res);
    matC[Astart + Bstart] = res;
}


__global__ void matmult_shared(CUDA_FLOAT *matA, CUDA_FLOAT *matB, CUDA_FLOAT *matC)
{
    int BX = blockIdx.x;
    int BY = blockIdx.y;
    int TX = threadIdx.x;
    int TY = threadIdx.y;
    
    CUDA_FLOAT CValue = 0;

    int Row = BY*BS + TY;
    int Col = BX*BS + TX;

    __shared__ CUDA_FLOAT As[BS][BS];
    __shared__ CUDA_FLOAT Bs[BS][BS];

    for (int k = 0; k < WA; k++) {

        //  if (k*BS + TX < WA && Row < HA)
        //      As[TY][TX] = matA[Row*WA + k*BS + TX];
        //  else
        //      As[TY][TX] = 0.0;

        //  if (k*BS + TY < HB && Col < WB)
        //      Bs[TY][TX] = matB[(k*BS + TY)*WB + Col];
        //  else
        //      Bs[TY][TX] = 0.0;

        As[TY][TX] = matA[Row * WA * BS + k*BS + TX];
        Bs[TY][TX] = matB[(k * BS + TY) * WB * BS + Col];

        __syncthreads();

        for (int n = 0; n < BS; ++n)
            CValue += As[TY][n] * Bs[n][TX];

        __syncthreads();
    }

    // if (Row < HC && Col < WC)
    matC[Row * WC * BS + Col] = CValue;
}


int main()
{
    CUDA_FLOAT *devMatA, *devMatB, *devMatC, *hostMatA, *hostMatB, *hostMatC;
    // allocate device memory
    cudaMalloc ( (void **) &devMatA, WA * BS * HA * BS * sizeof ( CUDA_FLOAT ) );
    cudaMalloc ( (void **) &devMatB, WB * BS * HB * BS * sizeof ( CUDA_FLOAT ) );
    cudaMalloc ( (void **) &devMatC, WC * BS * HC * BS * sizeof ( CUDA_FLOAT ) );
    hostMatA = (CUDA_FLOAT *)malloc(WA * BS * HA * BS * sizeof ( CUDA_FLOAT ));
    hostMatB = (CUDA_FLOAT *)malloc(WB * BS * HB * BS * sizeof ( CUDA_FLOAT ));
    hostMatC = (CUDA_FLOAT *)malloc(WC * BS * HC * BS * sizeof ( CUDA_FLOAT ));

    int i, j;
    for (i = 0; i < HA * BS; i++)
        for (j = 0; j < WA * BS; j++)
            hostMatA[WA * BS * i + j] = 1;

    for (i = 0; i < HB * BS; i++)
        for (j = 0; j < WB * BS; j++)
            hostMatB[WB * BS * i + j] = 1;

    cudaMemcpy ( devMatA, hostMatA, WA * BS * HA * BS * sizeof ( CUDA_FLOAT ), cudaMemcpyHostToDevice );
    cudaMemcpy ( devMatB, hostMatB, WB * BS * HB * BS * sizeof ( CUDA_FLOAT ), cudaMemcpyHostToDevice );

    matmult_shared<<<dim3(WC, HC), dim3(BS, BS)>>> (devMatA, devMatB, devMatC);
    // copy results from device to host memory
    // cudaMemcpy ( hostMatA, devMatA, WA * BS * HA * BS * sizeof ( CUDA_FLOAT ), cudaMemcpyDeviceToHost );
    // cudaMemcpy ( hostMatB, devMatB, WB * BS * HB * BS * sizeof ( CUDA_FLOAT ), cudaMemcpyDeviceToHost );
    cudaMemcpy ( hostMatC, devMatC, WC * BS * HC * BS * sizeof ( CUDA_FLOAT ), cudaMemcpyDeviceToHost );
    
    // for (i = 0; i < HC * BS; i++)
    // {
    //    for (j = 0; j < WC * BS; j++)
           // printf("%f ", hostMatC[WC * BS * i + j]);
        // printf("\n");
    // }

    cudaFree ( devMatA );
    cudaFree ( devMatB );
    cudaFree ( devMatC );
    free(hostMatA);
    free(hostMatB);
    free(hostMatC);
    
    return 1;
}
