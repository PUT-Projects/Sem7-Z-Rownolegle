
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>


#define BS 16

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, int blocks, int threads);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
     __shared__ int As[3];
    if (i <3) {As[i] = 0;}
    atomicAdd(&As[0], a[i] + b[i]);
    __syncthreads();
    atomicAdd(&As[1], a[i] + b[i]);
    As[2]+=a[i] + b[i];
    if (i==0) atomicAdd(&c[0], As[0] );
    if (i==1) atomicAdd(&c[1], As[1] );
    if (i==2) atomicAdd(&c[2], As[2] );
    __syncthreads();
    if (i==3) atomicAdd(&c[3], As[0] );
    if (i==2) atomicAdd(&c[4], As[1] );
    if (i==0) atomicAdd(&c[5], As[2] );
    __syncthreads();
     c[6]+=a[i] + b[i];
    __syncthreads();
    atomicAdd(&c[7], a[i] + b[i]);
// poczatkowo tablice a[] b[] zawieraja wartosci jednakowe = 1
// początkowow tablica c[] zawiera wartosci jednakowe = 0
// które wyniki w pamieci globalnej zostanie wyznaczone poprawnie ?
// jakie są przyczyny braku poprawności w obliczniu wyników w pamieci globalnej ?
// które synchronizacje __syncthreads nie są potrzebne ?
}

int main(int argc, char *argv[])
{
    if (argc != 4) {
        printf("Uzycie: %s N threads blocks\nWymagane:  N=threads >7  \n ", argv[0]);
        return 1;
    }

    int N       = atoi(argv[1]);
    int threads = atoi(argv[2]);
    int blocks  = atoi(argv[3]);

    const int arraySize = N;
     int a[arraySize];
     int b[arraySize];  
    for (int i = 0; i < arraySize; i++) a[i] = 1;//i;
    for (int i = 0; i < arraySize; i++) b[i] = 1;//i;

    int c[arraySize] = { 0};
    

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize,blocks,threads );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("wyniki  = {%d,%d,%d,%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4],c[5],c[6],c[7]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    //getchar();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, int blocks, int threads  )
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
 //   clock_t startt, stopt;
    float msecTotal = 0.0f;
 //   double gigaFlops = 0;
    
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

   cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
     cudaStatus = cudaMemcpy(dev_c, c, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        goto Error;
    }
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        goto Error;
    }

    cudaStatus = cudaEventRecord(start, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed");
        goto Error;
    }
 
    addKernel<<<blocks, threads>>>(dev_c, dev_a, dev_b);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
       
    cudaStatus = cudaEventRecord(stop, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed");
        goto Error;
    }
   
    cudaStatus = cudaEventSynchronize(stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventSynchronize(stop) failed");
        goto Error;
    }
    
    cudaEventElapsedTime(&msecTotal, start, stop);
      printf(
        "Time= %.3f msec \n",msecTotal);
   
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
