#pragma once

////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
// #include <opencv2/xfeatures2d.hpp>
////////////////////////////////////////////////////////////////
#define DEBUG

/********************************************************
 * interface
 * *****************************************************/
struct Object
{
    int id;
    float confidence;
    cv::Rect bbox;
    std::vector<cv::Point> contour;
    float hit_high_response_score;
    float shadow_score;
    float contour_score;
};
struct AntClpAlgParam {
  int blur[2] = {5, 3};
  int segment_radius[2] = {3, 1};
  int enlarge_pixs[2] = {6, 3};
  int canny_radius[2] = {3, 1};
  int morph_size[2] = {5, 3};
};
class AntClpAlg
{

public:

    AntClpAlg();
    ~AntClpAlg();
    
    //API
    void SetMask(const cv::Mat& mask, int mode);
    int input_before(const cv::Mat& rawImg);
    int input_after(const cv::Mat& rawImg);
    int process(cv::Mat& resultImg,  void* userData);
    void align(const cv::Mat& src, const cv::Mat& dst, cv::Mat& out);
    bool IsShow = true;
private:
    //图像宽高
    int imageWidth_;
    int imageHeight_;
    std::vector<Object> DetectObjects();
    std::vector<Object> ObjectsProgress(const std::vector<Object>& objects);

    void Similarity(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask);
    float ContourSimilarity(const cv::Mat& img1, const cv::Mat& img2, std::string idx, cv::Rect boundingBox);
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

    void align_by_gray(const cv::Mat& src, const cv::Mat& dst, cv::Mat& out);

    void align_by_orb(const cv::Mat& src, const cv::Mat& dst, cv::Mat& out);

    void segmentation();
    double ShadowDetect(const cv::Rect& bbox, const cv::Mat& mask, cv::Mat& result_mask, std::string idx);
    bool IsSimilarPixels(const cv::Point& center, int radius, double low_threshold,
                                  double high_threshold, double& dist);
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
    //gray color space
    cv::Mat prevGrayImg_;
    cv::Mat currGrayImg_;
    // roi mode 0-big 1-small
    int mode_ = 0;  
    cv::Mat mask_;
    //mask image
    cv::Mat object_mask_;
    // similar
    cv::Mat low_response_mask_;
    cv::Mat high_response_mask_;
    cv::Mat similarity_mask_;
    float hit_high_response_threshold_ = 0.1;//越高越严格
    // 蓝光抑制
    bool blue_suppression_ = true;
    int blue_threshold1 = 150;
    int blue_threshold2 = 30;
    // 背景阴影抑制
    bool shadow_suppression_ = true;
    int shadow_h_;
    int shadow_s_;
    int shadow_v_ = 100;
    float shadow_threshold_ = 0.6;//越高越严格
    // 边缘检测
    int contour_low_thresh_= 50;
    int contour_high_thresh_ = 100;
    float contour_threshold_ = 0.15;
    std::vector<cv::Rect> bboxes_;

    float object_area_threshold_ = 40;
    // segment
    int segment_low_thresh_ = 35;
    int segment_high_thresh_ = 100;

    // about size
    int kernel_size_[2] = {7, 3};
    int blur_kernel_size_[2] = {5, 3};
    int segment_kernel_size_[2] = {kernel_size_[0], kernel_size_[1]};
    int contour_kernel_size_[2] = {kernel_size_[0], kernel_size_[1]};
    int morph_kernel_size_[2] = {kernel_size_[0], kernel_size_[1]}; 
    int padding_size_[2] = {(kernel_size_[0]-1)/2 + 3, (kernel_size_[1]-1)/2 + 2};

    // AntClpAlgParam param_;
    ////////////////////////////////////////////////////////////////
};



