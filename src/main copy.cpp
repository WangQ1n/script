#include "func_test.h"
#include "gpu_speed_test.h"
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <cmath>
#include<opencv2/opencv.hpp>
#include<unordered_map>
#include<map>
bool projectTest();

std::pair<int, int> FindStonePair(std::vector<float> stone_weights, float diff);
std::vector<std::pair<int, int>> FindStonePairs(std::vector<float> stone_weights, float diff);
unsigned int uniquePathsWithObstacles(std::vector<std::vector<int>>& obstacleGrid);
unsigned int uniquePathsWithObstacles2(std::vector<std::vector<int>>& obstacleGrid);

int main(int argc, char *argv)
{

  // projectTest();
  // std::vector<float> stones = {1,4,2,4,2,8,5,4,3,5,6};
  // float diff = 2;
  // std::vector<std::pair<int, int>> stone_pairs = FindStonePairs(stones, diff);
  // for (auto pair : stone_pairs) {
  //   printf("%d -- %d\n", pair.first, pair.second);
  // }
  // std::pair<int, int> stone_pair = FindStonePair(stones, diff);
  // printf ("%d -- %d \n", stone_pair.first, stone_pair.second);

  std::vector<std::vector<int>> obstacle_grid = {{0,0,0,0,1},
                                                 {0,0,0,1,0},
                                                 {1,0,0,0,0}};
  unsigned int paths = uniquePathsWithObstacles2(obstacle_grid);
  printf("total path:%d\n", paths);
  return 1;
}

bool projectTest()
{
  cv::FileStorage fsSettings("/home/tzrobot/data/wangqin/tzCompany/20230228/sensors_calib.yml", cv::FileStorage::READ);
  if (!fsSettings.isOpened())
  {
    return false;
  }
  cv::Mat UTM_to_cam;
  fsSettings["UTM_to_cam_111111_1"] >> UTM_to_cam;
  std::vector<cv::Point2f> vec_cam_uv;
  std::vector<cv::Point2f> vec_lidar_w;
  vec_lidar_w.emplace_back(cv::Point2f(7.247481, 53.765839));
  cv::perspectiveTransform(vec_lidar_w, vec_cam_uv, UTM_to_cam);
  cv::Point2i cam_uv = vec_cam_uv[0];
  if (cam_uv.x >= 0 && cam_uv.x < 1920 && cam_uv.y >= 0 && cam_uv.y < 1080) {
    printf("图像感知区域 \n");
  } else {
    printf("超出图像感知区域 \n");
  }
  return true;
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

/*
 * Problem 2 Find stone pair(s)
 */
/**
 * @brief  求解石子里重量相差为diff的一对配对索引
 * @param  stone_weights: 每个石子的重量
 * @param  diff: 石子间配对的重量差值
 * @return std::pair<int, int>: 一对配对的石子索引
 */
std::pair<int, int> FindStonePair(std::vector<float> stone_weights, float diff)
{
  std::pair<int, int> stone_pair;
  if (stone_weights.size() < 2)
    return stone_pair;

  std::unordered_map<float, int> map;
  for (size_t i = 0; i < stone_weights.size(); i++) {
    float weight = stone_weights[i];
    float up_value = weight + diff;
    float low_value = weight - diff;

    auto iter = map.find(up_value);
    if (iter != map.end()) {
      printf("%f: %f ->  %d \n", weight, iter->first, iter->second);
      stone_pair = std::make_pair(iter->second, i);
      break;
    }

    iter = map.find(low_value);
    if (iter != map.end()) {
      printf("%f: %f ->  %d \n", weight, iter->first, iter->second);
      stone_pair = std::make_pair(iter->second, i);
      break;
    }
    map.insert(std::make_pair(weight, i));
  }

  return stone_pair;
}
/*
 * 2.   空间复杂度O(n), 时间复杂度O(n)
*/

/**
 * @brief 求解石子里重量相差为diff的多对配对的索引
 * @param  stone_weights: 石子的重量
 * @param  diff: 石子间配对的重量差值
 * @return std::vector<std::pair<int, int>>: 所有配对的石子索引
 */
std::vector<std::pair<int, int>> FindStonePairs(std::vector<float> stone_weights, float diff)
{
  std::vector<std::pair<int, int>> stone_pairs;
  if (stone_weights.size() < 2)
    return stone_pairs;

  std::unordered_multimap<float, int> mmap;
  for (size_t i = 0; i < stone_weights.size(); i++) {
    float weight = stone_weights[i];
    float up_value = weight + diff;
    float low_value = weight - diff;

    auto iter = mmap.equal_range(up_value);
    for (auto it = iter.first; it != iter.second; it++) {
      // printf("%f: %f ->  %d \n", weight, it->first, it->second);
      stone_pairs.emplace_back(std::make_pair(it->second, i));
    }

    iter = mmap.equal_range(low_value);
    for (auto it = iter.first; it != iter.second; it++) {
      // printf("%f: %f ->  %d \n", weight, it->first, it->second);
      stone_pairs.emplace_back(std::make_pair(it->second, i));
    }
    mmap.insert(std::make_pair(weight, i));
  }

  return stone_pairs;
}

/*
 * 2.   空间复杂度O(n), 时间复杂度O(max(R, N))
*/


/**
 * Problem 1 Rabbit goes home
 */
// QuestionA
unsigned int uniquePathsWithObstacles(std::vector<std::vector<int>>& obstacleGrid) {
  int m = obstacleGrid.size();
  int n = obstacleGrid[0].size();
	if (obstacleGrid[m - 1][n - 1] == 1 || obstacleGrid[0][0] == 1) //如果在起点或终点出现了障碍，直接返回0
    return 0;
  std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));
  for (int i = 0; i < m && obstacleGrid[i][0] == 0; i++) {
    dp[i][0] = 1;
  }
  for (int j = 0; j < n && obstacleGrid[0][j] == 0; j++) {
    dp[0][j] = 1;
  }
  for (int i = 1; i < m; i++) {
    for (int j = 1; j < n; j++) {
      if (obstacleGrid[i][j] == 1) 
        continue;

      dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
    }
  }
  return dp[m - 1][n - 1];
}

// QuestionB
unsigned int uniquePathsWithObstacles2(std::vector<std::vector<int>>& obstacleGrid) {
  if (obstacleGrid[0][0] == 1)
      return 0;
  std::vector<int> dp(obstacleGrid[0].size());
  if (obstacleGrid[0].size() > obstacleGrid.size()) {
    dp.resize(obstacleGrid.size());
    for (int j = 0; j < dp.size(); ++j) {
      if (obstacleGrid[j][0] == 1)
        dp[j] = 0;
      else if (j == 0)
        dp[j] = 1;
      else
        dp[j] = dp[j-1];
    }
    for (int i = 1; i < obstacleGrid[0].size(); ++i) {
      for (int j = 0; j < dp.size(); ++j){
        if (obstacleGrid[j][i] == 1)
          dp[j] = 0;
        else if (j != 0)
          dp[j] = dp[j] + dp[j-1];
      }
    }
  } else {
    for (int j = 0; j < dp.size(); ++j) {
      if (obstacleGrid[0][j] == 1)
        dp[j] = 0;
      else if (j == 0)
        dp[j] = 1;
      else
        dp[j] = dp[j-1];
    }
    for (int i = 1; i < obstacleGrid.size(); ++i) {
      for (int j = 0; j < dp.size(); ++j){
        if (obstacleGrid[i][j] == 1)
          dp[j] = 0;
        else if (j != 0)
          dp[j] = dp[j] + dp[j-1];
      }
    }
  }

  return dp.back();
}
