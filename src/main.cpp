#include "func_test.h"
#include "gpu_speed_test.h"
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <cmath>
#include<opencv2/opencv.hpp>
int main(int argc, char *argv)
{
    double yaw = 356;
    double yaw_bak = yaw;
    yaw = yaw*M_PI/180.0;
    //0 ~ 2pi 逆时针   正北 0
    yaw  = 2.0*M_PI - yaw ;
    if (yaw  >= 2.0*M_PI)
      yaw  = 0;

    //0 ~ 2pi 逆时针   正东 0
    yaw += 0.5*M_PI;
    if (yaw >= 2.0*M_PI)
      yaw -= 2.0*M_PI;

    cv::Point end;
    cv::Point p_draw = cv::Point(50, 50);
    end.x = p_draw.x +  15*cos(yaw);
    end.y = p_draw.y +  15*sin(yaw);

    std::cout << "ori_yaw:" << yaw_bak << " -> " << yaw << std::endl;
    std::cout << "ori_point:" << p_draw << " -> " << end << std::endl;
    // GpuInfos();
    // system("pause");
    // while(1)
    // {
    //     GpuSpeedTest();

    // }
    
    return 1;
}

void GpuSpeedTest()
{

    int size = 24883200;
    int *array = (int *) malloc(sizeof(int) * size);
    std::clock_t startTime, endTime;

    setInput(array, size);

    startTime = std::clock();
    Run(array, size);
    endTime = std::clock();
    std::cout << "The Gpu time: " << size << "/" << (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000<< "ms" << std::endl;
    
    GPufree();
    free(array);
}

void GpuInfos()
{
  int dev = 0;
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, dev);
  std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
  printf("Grid Size: (%d, %d, %d)\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
  std::cout << "SM的数量:" << devProp.multiProcessorCount << std::endl;
  std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
  std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
  std::cout << "SM最大线程数:" << devProp.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "SM最大线程纬度:" << devProp.maxThreadsDim << std::endl;

}