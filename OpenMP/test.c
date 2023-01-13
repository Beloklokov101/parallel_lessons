#include <stdio.h>
#include <omp.h>
int main(int argc, char *argv[])
{
    int A[10], B[10], C[10], i, n;
    for (i=0; i<10; i++)
    {
        A[i]=i; 
        B[i]=2*i; 
        C[i]=0;
    }

    #pragma omp parallel for shared(A,B,C) private(i)
    for (i=0; i<10; i++){
        C[i] = A[i] + B[i];
        printf("Task %d element %d\n", omp_get_thread_num(), i);
    }
}