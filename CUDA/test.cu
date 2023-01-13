#include <cuda.h>
#include <stdio.h>

__global__ void test_kern(int *a){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[idx] = idx;
//   printf("--%d-- %d %d\n", idx, blockIdx.x, threadIdx.x);
  return;
}

int main(){
  int i;
  int a[9];
  int *a_d;
  cudaMalloc(&a_d, 10*sizeof(int));
  test_kern<<<dim3(3), dim3(3)>>> (a_d);
  cudaMemcpy(a, a_d, 9 * sizeof(int),  cudaMemcpyDeviceToHost);
  for (i = 0; i < 9 ; i++){
    printf("%d\n", a[i]);  
  }
  return 1;
}