#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
// 黑 灰 白 红 橙 黄 绿 青 蓝 紫
std::vector<cv::Scalar> colors_ = {
    cv::Scalar(0, 0, 0),       cv::Scalar(110, 110, 110),
    cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 255),
    cv::Scalar(0, 170, 230),   cv::Scalar(0, 228, 230),
    cv::Scalar(0, 255, 0),     cv::Scalar(220, 230, 0),
    cv::Scalar(255, 0, 0),     cv::Scalar(230, 0, 220)};

std::vector<std::string> colors_text_ = {"黑", "灰", "白", "红", "橙",
                                         "黄", "绿", "青", "蓝", "紫"};

int classifyColor(int H, int S, int V);

cv::Mat SiftFeature(cv::Mat img1);

cv::Mat guss(cv::Ptr<cv::BackgroundSubtractorMOG2> bg, cv::Mat img);

cv::Mat GMG(cv::Ptr<cv::BackgroundSubtractorKNN> bg, cv::Mat img);