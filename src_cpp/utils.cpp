#include "utils.h"

int classifyColor(int H, int S, int V) {
  if (H >= 0 && H <= 180 && S >= 0 && S <= 255 && V >= 0 && V < 46) {
    return 0;
  } else if (H >= 0 && H <= 180 && S >= 0 && S < 53 && V >= 46 && V <= 220) {
    return 1;
  } else if (H >= 0 && H <= 180 && S >= 0 && S < 43 && V >= 221 && V <= 255) {
    return 2;
  } else if (((H >= 0 && H <= 10) || (H >= 156 && H <= 180)) && S >= 43 &&
             S <= 255 && V >= 46 && V <= 255) {
    return 3;
  } else if (H >= 11 && H <= 25 && S >= 43 && S <= 255 && V >= 46 && V <= 255) {
    return 4;
  } else if (H >= 26 && H <= 34 && S >= 43 && S <= 255 && V >= 46 && V <= 255) {
    return 5;
  } else if (H >= 35 && H <= 77 && S >= 43 && S <= 255 && V >= 46 && V <= 255) {
    return 6;
  } else if (H >= 78 && H <= 99 && S >= 43 && S <= 255 && V >= 46 && V <= 255) {
    return 7;
  } else if (H >= 100 && H <= 124 && S >= 43 && S <= 255 && V >= 46 &&
             V <= 255) {
    return 8;
  } else if (H >= 125 && H <= 155 && S >= 43 && S <= 255 && V >= 46 &&
             V <= 255) {
    return 9;
  } else {
    return -1;
  }
}

cv::Mat SiftFeature(cv::Mat img1) {
  // 创建 SIFT 特征提取器
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

  // 检测关键点和计算描述符
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);


  // 在图像上绘制关键点
  cv::Mat image_with_keypoints1;
  cv::drawKeypoints(img1, keypoints1, image_with_keypoints1);

  // 显示图像和关键点
  cv::imshow("Image with Keypoints1", image_with_keypoints1);
  return image_with_keypoints1;
}

cv::Mat guss(cv::Ptr<cv::BackgroundSubtractorMOG2> bg, cv::Mat img) {
  cv::Mat mask;
  bg->apply(img, mask);
  return mask;
}

cv::Mat GMG(cv::Ptr<cv::BackgroundSubtractorKNN> bg, cv::Mat img) {
  cv::Mat mask;
  bg->apply(img, mask);
  return mask;
}