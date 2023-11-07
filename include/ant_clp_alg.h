#ifndef  __ANT_CLP_ALG_H__
#define __ANT_CLP_ALG_H__

////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
////////////////////////////////////////////////////////////////


/********************************************************
 * interface
 * *****************************************************/
class AntClpAlg
{
public:

    AntClpAlg();
    ~AntClpAlg();
    
    //API
    int input_before(const cv::Mat& rawImg);
    int input_after(const cv::Mat& rawImg);
    int process(cv::Mat& resultImg,  void* userData);

private:
    //图像宽高
    int imageWidth_;
    int imageHeight_;

    //
    void pre_process(const cv::Mat& inImg, cv::Mat& outImg);
    //
    void image_segmentation(const cv::Mat& Img1, const cv::Mat& Img2, cv::Mat& maskImg);
    //
    void gradient_exaction(const cv::Mat& inImg, cv::Mat& outImg);
    //
    void morphphological_process(cv::Mat& maskImg);
    //
    void advance_process(cv::Mat& Img1, cv::Mat& Img2, cv::Mat& maskImg);
    //
    void find_objdect(const cv::Mat& inImg, std::vector<cv::Rect>& vBoundRect);
    //
    void box_filter(std::vector<cv::Rect>& vBoundRect, std::vector<int>& picked, float nms_threshold);


    ////////////////////////////////////////////////////////////////
    //src image
    cv::Mat prevSrcImg_;
    cv::Mat currSrcImg_;
    //Pre-Processing image
    cv::Mat prevPreProImg_;
    cv::Mat currPreProImg_;
    //feature image
    cv::Mat prevFtImg_;
    cv::Mat currFtImg_;
    //hsv color space
    cv::Mat prevHsvImg_;
    cv::Mat currHsvImg_;
    //mask image
    cv::Mat maskImg_;
    ////////////////////////////////////////////////////////////////
    bool IsShow = true;
};

struct Point {
    int x;
    int y;
    Point(int _x, int _y) : x(_x), y(_y) {}
};

void hsvRegionGrowing(cv::Mat& inputImage, cv::Mat& stdImage, cv::Mat& outputImage, int x, int y, cv::Vec3i seedColor, int hueThreshold, int saturationThreshold, int valueThreshold) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    if (x < 0 || x >= cols || y < 0 || y >= rows || outputImage.at<uchar>(y, x) == 0) {
        return;
    }

    // int hueDiff = std::abs(seedColor[0] - inputImage.at<cv::Vec3b>(y, x)[0]);
    // int saturationDiff = std::abs(seedColor[1] - inputImage.at<cv::Vec3b>(y, x)[1]);
    // int valueDiff = std::abs(seedColor[2] - inputImage.at<cv::Vec3b>(y, x)[2]);
    cv::Vec3i currentColor = inputImage.at<cv::Vec3b>(y, x);
    cv::Vec3i stdColor = stdImage.at<cv::Vec3b>(y, x);
    
    int hDiff = std::min(abs(currentColor[0] - stdColor[0]), 180 - abs(currentColor[0] - stdColor[0]));
    int smin = abs(currentColor[1] - stdColor[1]);
    int vmin = abs(currentColor[2] - stdColor[2]);
    // if (hDiff > 20 || vmin > MAX(currentColor[2], stdColor[2]) * 0.7 || smin > 100)
    //     return;
    // if (vmin > MAX(currentColor[2], stdColor[2]) * 0.4 && smin < vmin * 0.1)
    //     return;
    currentColor[0] *= 0.5;
    currentColor[1] *= 0.3;
    currentColor[2] *= 0.2;
    int colorDiff = cv::norm(currentColor - seedColor, cv::NORM_L2);
    // stdColor[0] *= 0.5;
    // stdColor[1] *= 0.0;
    // stdColor[2] *= 0.0;
    // cv::Vec3i tmp = currentColor;
    // tmp[1] *= 0.0;
    // tmp[2] *= 0.0;
    // int stdColorDiff = cv::norm(tmp - stdColor, cv::NORM_L2);

    // if (hueDiff <= hueThreshold && saturationDiff <= saturationThreshold && valueDiff <= valueThreshold) {
    // if ((hDiff < 2) || (colorDiff <= hueThreshold)) {
    if ((colorDiff <= hueThreshold)) {
        // printf ("pos(%d, %d) diff:%d", x, y, colorDiff);
        // std::cout <<  currentColor - seedColor<< "-" << currentColor << "-" << seedColor << std::endl;
        outputImage.at<uchar>(y, x) = 0;

        hsvRegionGrowing(inputImage, stdImage, outputImage, x - 1, y, currentColor, hueThreshold, saturationThreshold, valueThreshold);
        hsvRegionGrowing(inputImage, stdImage, outputImage, x + 1, y, currentColor, hueThreshold, saturationThreshold, valueThreshold);
        hsvRegionGrowing(inputImage, stdImage, outputImage, x, y - 1, currentColor, hueThreshold, saturationThreshold, valueThreshold);
        hsvRegionGrowing(inputImage, stdImage, outputImage, x, y + 1, currentColor, hueThreshold, saturationThreshold, valueThreshold);
    }
}

void colorRegionGrowing(cv::Mat& inputImage, cv::Mat& outputImage, int seedX, int seedY, int colorThreshold) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    std::queue<Point> pointsQueue;
    pointsQueue.push(Point(seedX, seedY));
    cv::Vec3b seedColor = inputImage.at<cv::Vec3b>(seedY, seedX);

    while (!pointsQueue.empty()) {
        Point currentPoint = pointsQueue.front();
        pointsQueue.pop();

        int x = currentPoint.x;
        int y = currentPoint.y;

        if (x < 0 || x >= cols || y < 0 || y >= rows || outputImage.at<uchar>(y, x) == 0) {
            continue;
        }

        cv::Vec3b currentColor = inputImage.at<cv::Vec3b>(y, x);
        int colorDiff = cv::norm(currentColor - seedColor, cv::NORM_L2);
        std::cout << "colorDiff:" << colorDiff << "  ";
        if (colorDiff <= colorThreshold) {
            outputImage.at<uchar>(y, x) = 0;

            pointsQueue.push(Point(x - 1, y));
            pointsQueue.push(Point(x + 1, y));
            pointsQueue.push(Point(x, y - 1));
            pointsQueue.push(Point(x, y + 1));
        }
    }
    std::cout << std::endl;
}

cv::Mat SiftFeature(cv::Mat img1, cv::Mat img2) {
  // 转换为灰度图像
//   cv::Mat grayImg;
//   cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
//   cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);
  // 创建 SIFT 特征提取器
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

  // 检测关键点和计算描述符
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
  sift->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
  // 创建 Brute-Force 匹配器
  cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create();

  // 进行描述符匹配
  std::vector<cv::DMatch> matches;
  matcher->match(descriptors1, descriptors2, matches);

  // 绘制匹配结果
  cv::Mat match_image;
  cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, match_image);
  float avg = 0;
  for (auto item : matches) {
    avg = (avg + item.distance) / 2;
  }
  printf("len: %d, %d, diff: %d, avg: %f\n", keypoints1.size(), keypoints2.size(), keypoints1.size() - keypoints2.size(), avg);
  // 显示匹配结果
  cv::imshow("Feature Matches", match_image);

  // 在图像上绘制关键点
  cv::Mat image_with_keypoints1;
  cv::drawKeypoints(img1, keypoints1, image_with_keypoints1);
  cv::Mat image_with_keypoints2;
  cv::drawKeypoints(img2, keypoints2, image_with_keypoints2);

  // 显示图像和关键点
  cv::imshow("Image with Keypoints1", image_with_keypoints1);
  cv::imshow("Image with Keypoints2", image_with_keypoints2);
  return match_image;
}
#endif


