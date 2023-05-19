#include<vector>
#include<unordered_map>
#include<map>

std::pair<int, int> FindStonePair(std::vector<float>& stone_weights, float diff);
std::vector<std::pair<int, int>> FindStonePairs(std::vector<float>& stone_weights, float diff);
unsigned int uniquePathsWithObstacles(std::vector<std::vector<int>>& obstacle_grid);
unsigned int uniquePathsWithObstaclesBySpaceOptimization(std::vector<std::vector<int>>& obstacle_grid);

int main(int argc, char *argv)
{

  std::vector<float> stones = {1,4,2,3,2,8,5,4,3,5,6};
  float diff = 2;
  std::pair<int, int> stone_pair = FindStonePair(stones, diff);
  printf ("%d -- %d \n", stone_pair.first, stone_pair.second);
  std::vector<std::pair<int, int>> stone_pairs = FindStonePairs(stones, diff);
  for (auto pair : stone_pairs) {
    printf("%d -- %d\n", pair.first, pair.second);
  }

  std::vector<std::vector<int>> obstacle_grid = {{0,0,0,0,1},
                                                 {0,0,0,1,0},
                                                 {1,0,0,0,0}};
  unsigned int paths = uniquePathsWithObstacles(obstacle_grid);
  printf("total path:%d\n", paths);
  paths = uniquePathsWithObstaclesBySpaceOptimization(obstacle_grid);
  printf("total path:%d\n", paths);
  return 1;
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
std::pair<int, int> FindStonePair(std::vector<float>& stone_weights, float diff)
{
  std::pair<int, int> stone_pair(-1,-1);
  if (stone_weights.size() < 2)
    return stone_pair;

  std::unordered_map<float, int> map;
  for (size_t i = 0; i < stone_weights.size(); i++) {
    float weight = stone_weights[i];
    float up_value = weight + diff;
    float low_value = weight - diff;

    auto iter = map.find(up_value);
    if (iter != map.end()) {
      // printf("%f: %f ->  %d \n", weight, iter->first, iter->second);
      stone_pair = std::make_pair(iter->second, i);
      break;
    }

    iter = map.find(low_value);
    if (iter != map.end()) {
      // printf("%f: %f ->  %d \n", weight, iter->first, iter->second);
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
std::vector<std::pair<int, int>> FindStonePairs(std::vector<float>& stone_weights, float diff)
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
unsigned int uniquePathsWithObstacles(std::vector<std::vector<int>>& obstacle_grid) {
  int m = obstacle_grid.size();
  int n = obstacle_grid[0].size();
	if (obstacle_grid[m - 1][n - 1] == 1 || obstacle_grid[0][0] == 1) //如果在起点或终点出现了障碍，直接返回0
    return 0;
  std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));
  for (int i = 0; i < m && obstacle_grid[i][0] == 0; i++) {
    dp[i][0] = 1;
  }
  for (int j = 0; j < n && obstacle_grid[0][j] == 0; j++) {
    dp[0][j] = 1;
  }
  for (int i = 1; i < m; i++) {
    for (int j = 1; j < n; j++) {
      if (obstacle_grid[i][j] == 1) 
        continue;

      dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
    }
  }
  return dp[m - 1][n - 1];
}
/*
 * 空间复杂度O(M*N), 时间复杂度O(M*N)
 */

// QuestionB
unsigned int uniquePathsWithObstaclesBySpaceOptimization(std::vector<std::vector<int>>& obstacle_grid) {
  if (obstacle_grid[0][0] == 1)
      return 0;
  std::vector<int> dp(obstacle_grid[0].size());
  if (obstacle_grid[0].size() > obstacle_grid.size()) {
    dp.resize(obstacle_grid.size());
    for (int j = 0; j < dp.size(); ++j) {
      if (obstacle_grid[j][0] == 1)
        dp[j] = 0;
      else if (j == 0)
        dp[j] = 1;
      else
        dp[j] = dp[j-1];
    }
    for (int i = 1; i < obstacle_grid[0].size(); ++i) {
      for (int j = 0; j < dp.size(); ++j){
        if (obstacle_grid[j][i] == 1)
          dp[j] = 0;
        else if (j != 0)
          dp[j] = dp[j] + dp[j-1];
      }
    }
  } else {
    for (int j = 0; j < dp.size(); ++j) {
      if (obstacle_grid[0][j] == 1)
        dp[j] = 0;
      else if (j == 0)
        dp[j] = 1;
      else
        dp[j] = dp[j-1];
    }
    for (int i = 1; i < obstacle_grid.size(); ++i) {
      for (int j = 0; j < dp.size(); ++j){
        if (obstacle_grid[i][j] == 1)
          dp[j] = 0;
        else if (j != 0)
          dp[j] = dp[j] + dp[j-1];
      }
    }
  }

  return dp.back();
}
/*
 * 空间复杂度O(min(M, N)), 时间复杂度O(M*N)
 */