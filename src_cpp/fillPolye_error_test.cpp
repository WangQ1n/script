#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
int main() {
  // 文件路径
  std::string filepath = "../data/data.txt";

  // 打开文件流
  std::ifstream file(filepath);

  // 检查文件是否成功打开
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filepath << std::endl;
    return 1;
  }

  std::vector<cv::Point> contour;
  // 逐行读取文件内容
  std::string line;
  while (std::getline(file, line)) {
    // 处理每一行的数据，这里简单输出
    // std::cout << line << std::endl;
    if (line != "") {
      std::istringstream inputStream(line);
      int number1, number2;
      // 从字符串流中读取整数
      char comma, semicolon;  // 用于处理逗号和分号
      inputStream >> std::ws >> number1 >> std::ws >> comma >> std::ws >> number2 >> std::ws >> semicolon;
      // 检查解析是否成功
      if (inputStream.fail() || comma != ',' || semicolon != ';') {
        std::cerr << "Error parsing input string." << std::endl;
        return 1;
      }
      //   std::cout << "Number : " << cv::Point(number1, number2) << std::endl;
      contour.emplace_back(cv::Point(number1, number2));
    }
  }
  //[391 x 726 from (650, 353)]
  int max_x = -999, max_y = -999, min_x = 999, min_y = 999;
  cv::Rect bbox = cv::boundingRect(contour);
  for (size_t i = 0; i < contour.size(); i++) {
    int diff_x = contour[i].x - bbox.x;
    int diff_y = contour[i].y - bbox.y;
    if (diff_x <= 0 || diff_y <= 0) {
      std::cout << contour[i] << "--" << bbox << std::endl;
    }
    if (diff_x > max_x) {
        max_x = diff_x;
    }
    if (diff_x < min_x) {
        min_x= diff_x;
    }
        if (diff_y > max_y) {
        max_y = diff_y;
    }
    if (diff_y < min_y) {
        min_y= diff_y;
    }
  }
  printf("%d, %d, %d, %d", min_x, min_y, max_x, max_y);
  cv::Mat mask = cv::Mat::zeros(bbox.height, bbox.width, CV_8UC1);
  std::vector<std::vector<cv::Point>> contours;
  contours.emplace_back(contour);
  cv::fillPoly(mask, std::vector<std::vector<cv::Point>>(1, contour), 255, 8, 0, cv::Point(-bbox.x, -bbox.y));
//   cv::fillConvexPoly(mask, contour, 255, 8, 0, )
  cv::imshow("show", mask);
  cv::waitKey(0);
  return 0;
}