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

__global__ void preprocess_and_concat_kernel(
  const uint8_t* __restrict__ img1, // HWC, BGR, uint8
  const uint8_t* __restrict__ img2,
  float* __restrict__ output,       // CHW, float32, C=6, H=224, W=224
  int inH, int inW, int outH, int outW, const float* __restrict__ mean, const float* __restrict__ std)
{
  int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_x = blockIdx.x * blockDim.x + threadIdx.x;

  if (dst_y >= outH || dst_x >= outW) return;

  float scale_x = static_cast<float>(inW) / outW;
  float scale_y = static_cast<float>(inH) / outH;

  // Output: 6 x H x W
  // Layout: [0-2]=img1(R,G,B), [3-5]=img2(R,G,B)

  for (int img_idx = 0; img_idx < 2; ++img_idx) {
      const uint8_t* img = (img_idx == 0) ? img1 : img2;

      float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
      float src_y = (dst_y + 0.5f) * scale_y - 0.5f;

      int x1 = floorf(src_x);
      int y1 = floorf(src_y);
      int x2 = min(x1 + 1, inW - 1);
      int y2 = min(y1 + 1, inH - 1);

      float dx = src_x - x1;
      float dy = src_y - y1;

      for (int c = 0; c < 3; ++c) {
          int bgr = 2 - c; // BGR → RGB

          float p1 = img[(y1 * inW + x1) * 3 + bgr];
          float p2 = img[(y1 * inW + x2) * 3 + bgr];
          float p3 = img[(y2 * inW + x1) * 3 + bgr];
          float p4 = img[(y2 * inW + x2) * 3 + bgr];

          float val = (1 - dx) * (1 - dy) * p1 +
                      dx * (1 - dy) * p2 +
                      (1 - dx) * dy * p3 +
                      dx * dy * p4;

          val = (val - mean[c]) / std[c];

          int out_c = img_idx * 3 + c; // Output channel index
          int out_index = out_c * outH * outW + dst_y * outW + dst_x;
          output[out_index] = val;
      }
  }
}

void preprocess(const uint8_t* img1, const uint8_t* img2, float* output, int srcH, int srcW, int dstH, int dstW, 
  const float* mean, const float* std, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((224 + block.x - 1) / block.x, (224 + block.y - 1) / block.y);

  preprocess_and_concat_kernel<<<grid, block, 0, stream>>>(
    img1, img2, output,
    srcH, srcW, dstH, dstW, mean, std);

}