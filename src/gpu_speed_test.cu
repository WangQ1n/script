#include "gpu_speed_test.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>

#define MAX_SHARED_SIZE 2048
#define BLOCK_SIZE_X 1024
inline void gassert(cudaError_t err_code, const char *file, int line)
{
  if (err_code != cudaSuccess)
  {
    fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(err_code), file, line);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) gassert(val, __FILE__, __LINE__)

static int *_array;

extern "C" __global__ void change(int *array, int size, int size2, int size3,int size4,int size5,int size6,int size7,int size8)
{
    // int index  = threadIdx.x + blockIdx.x * blockDim.x;
    // array[index] = 4*10+5;
}

void setInput(int *array, int size)
{
    checkCudaErrors(cudaMalloc(&_array, size * sizeof(int)));
    checkCudaErrors(cudaMemcpy(_array, array, size * sizeof(int), cudaMemcpyHostToDevice));


}
void Run(int *array, int size)
{
    // int *_array;

    int block_x, grid_x;
    block_x = (size > BLOCK_SIZE_X) ? BLOCK_SIZE_X : size;
    grid_x = (size - 1) / block_x + 1;

    change << < grid_x, block_x >> > (_array, size,2,3,4,5,6,7,8);
    checkCudaErrors(cudaDeviceSynchronize());

}

void GPufree()
{
    checkCudaErrors(cudaFree(_array));

}