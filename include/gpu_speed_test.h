#ifndef GPU_SPEED_TEST_H
#define GPU_SPEED_TEST_H
#include <opencv2/opencv.hpp>
void setInput(int *array, int size);
void GPufree();

void Run(int *array, int size);
void preprocess(const uint8_t* img1, const uint8_t* img2, float* output, int srcH, int srcW, int dstH, int dstW, 
  const float* mean, const float* std, cudaStream_t stream);
#endif