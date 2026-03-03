#include "ant_clp_alg.h"

#include <chrono>
#include <iostream>

#include "utils.h"
/********************************************************
 * 算法接口
 *
 * 输入第一张图像
 * input_before(const cv::Mat& rawImg)
 * 输入第二张图像
 * input_after(const cv::Mat& rawImg)
 * 获取检测结果 检测图像，目标数（默认Int类型）
 * process(cv::Mat& resultImg, void* userData)
 *
 *
 * TODO
 * 阴影，光斑检测过滤
 * *****************************************************/

// 八邻域偏移
const int xOffset[9] = {0, -1, 0, 1, 1, 1, 0, -1, -1};
const int yOffset[9] = {0, -1, -1, -1, 0, 1, 1, 1, 0};

// yolo nms
#define NMS_THRESH 0.45

//--------------------------------------------------------------//
// 最小距离阈值 （自适应TODO）
#define DEFAULT_COLOR_DISTANCE (20)
// 匹配数（最小）
#define DEFAULT_MATCH_NUM (1)
//--------------------------------------------------------------//

// 阴影检测  （自适应TODO）
#define DEFAULT_HUE_THVAL (10)
#define DEFAULT_SATURATION_THVAL (30)
#define DEFAULT_VALUE_THVAL (30)

#define DEFAULT_VALUE_THVAL_L (0.4)
#define DEFAULT_VALUE_THVAL_H (0.9)

#ifdef DEBUG
#define DEBUG_SHOW(winname, image) \
  { cv::imshow(winname, image); }
#else
#define DEBUG_SHOW(winname, image) \
  {}
#endif
// param
// int blur[] = {5, 3};
// int segment_radius[] = {5, 2};
// int segment_low_threshold[] = {35, 35};
// int segment_high_threshold[] = {100, 100};
// int enlarge_pixs[] = {10, 5};
// int canny_radius[] = {5, 2};
// int morph_size[] = {5, 3};
// int color_threshold[] = {50, 30};
// int threq = 0;
/**
 * @brief
 * @param
 * */
AntClpAlg::AntClpAlg() {
  //...
}

/**
 * @brief
 * @param
 * */
AntClpAlg::~AntClpAlg() {
  //...
}

void AntClpAlg::SetMask(const cv::Mat& mask, int mode) {
  mask_ = mask.clone();
  mode_ = mode;
  // cv::imshow("roi", roi);
}
/**
 * @brief input src image
 * @param rawImg 原始图像 BGR
 * */
int AntClpAlg::input_before(const cv::Mat& rawImg) {
  // check
  if (rawImg.empty()) {
    std::cout << "invalid input" << std::endl;
    return -1;
  }
  // copy src
  rawImg.copyTo(prevSrcImg_);
  // get width height TODO
  imageWidth_ = prevSrcImg_.cols;
  imageHeight_ = prevSrcImg_.rows;
  pre_process(prevSrcImg_, prevPreProImg_);
  // prevPreProImg_ = prevSrcImg_.clone();

  cv::cvtColor(prevPreProImg_, prevHsvImg_, cv::COLOR_BGR2HSV);

  cv::cvtColor(prevPreProImg_, prevGrayImg_, cv::COLOR_BGR2GRAY);
  // pre
  return 0;
}

/**
 * @brief input src image
 * @param rawImg 原始图像 BGR
 * */
int AntClpAlg::input_after(const cv::Mat& rawImg) {
  // check
  if (rawImg.empty()) {
    std::cout << "invalid input" << std::endl;
    return -1;
  }
  // copy src
  rawImg.copyTo(currSrcImg_);
  // pre
  pre_process(currSrcImg_, currPreProImg_);
  // currPreProImg_ = currSrcImg_.clone();
  cv::cvtColor(currPreProImg_, currHsvImg_, cv::COLOR_BGR2HSV);

  cv::cvtColor(currPreProImg_, currGrayImg_, cv::COLOR_BGR2GRAY);

  return 0;
}

/**
 * @brief pre_process 图像预处理函数 灰度转换与滤波
 * @param inImg 输入图像
 * @param outImg 输出图像 灰度
 * */
void AntClpAlg::pre_process(const cv::Mat& inImg, cv::Mat& outImg) {
  if (inImg.empty()) {
    std::cout << "PreProcess: input Image can not be empty" << std::endl;
    return;
  }
  //
  if (outImg.empty()) {
    outImg.create(inImg.size(), CV_8UC1);
  }
  cv::blur(inImg, outImg, cv::Size(blur_kernel_size_[mode_], blur_kernel_size_[mode_]));
}
//////////////////////////////////////////////////////////////////
void ContourDiff(const cv::Mat& img1, const cv::Mat& img2) {
  // 二值化蓝色通道
  cv::Mat thresh;
  // cv::threshold(img1, thresh, 128, 255, cv::THRESH_BINARY);
  cv::Canny(img1, thresh, 100, 200);  // 边缘检测，阈值可以调整

  // 进行轮廓检测
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  // 在原始图像上绘制轮廓
  cv::Mat contourImg = img2.clone();
  cv::drawContours(contourImg, contours, -1, cv::Scalar(0, 0, 255), 2);  // 红色绘制

  // 二值化蓝色通道
  cv::Mat thresh2;
  // cv::threshold(img2, thresh2, 128, 255, cv::THRESH_BINARY);
  cv::Canny(img2, thresh2, 100, 200);  // 边缘检测，阈值可以调整
  // 进行轮廓检测
  // std::vector<std::vector<cv::Point>> contours2;
  // std::vector<cv::Vec4i> hierarchy2;
  // cv::findContours(thresh2, contours2, hierarchy2, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  // 在原始图像上绘制轮廓
  // cv::Mat contourImg2 = img2.clone();
  // cv::drawContours(contourImg2, contours2, -1, cv::Scalar(0, 0, 255), 2);  // 红色绘制

  // if (!contours.empty() && !contours2.empty()) {
  //   double similarity = cv::matchShapes(contours[0], contours2[0], cv::CONTOURS_MATCH_I1, 0.0);
  //   std::cout << "contour similaritu: " << similarity << std::endl;
  // }
  // cv::imshow("img1 gray Channel", img1);
  cv::imshow("img1 thre Channel", thresh);
  // cv::imshow("Contours1", contourImg);

  // cv::imshow("img2 gray Channel", img2);
  cv::imshow("img2 thre Channel", thresh2);
  // cv::imshow("Contours2", contourImg2);
}

// bool isBlack(const cv::Vec3b& prev_color)
// 灰色背景阴影抑制
double AntClpAlg::ShadowDetect(const cv::Rect& bbox, const cv::Mat& mask, cv::Mat& result_mask,
                               std::string idx) {
  bool is_shadow = false;
  cv::Mat prev_hsv, curr_hsv;
  result_mask.create(bbox.size(), CV_8UC1);
  result_mask.setTo(cv::Scalar::all(123));
  prev_hsv = prevHsvImg_(bbox);
  curr_hsv = currHsvImg_(bbox);
  int light_pixs = 0;
  int dark_pixs = 0;
  int radius = (segment_kernel_size_[mode_] - 1) / 2;
  for (size_t j = 0; j < prev_hsv.rows; j++) {
    for (size_t i = 0; i < prev_hsv.cols; i++) {
      cv::Point center(i, j);
      if (mask_.at<uchar>(bbox.y + j, bbox.x + i) == 0 || mask.at<uchar>(j, i) == 0) {
        continue;
      }
      cv::Vec3b prev_color = prev_hsv.at<cv::Vec3b>(center);
      if (prev_color[1] > 43) {
        continue;
      }

      // 限制范围防止越界
      int startX = std::max(0, center.x - radius);
      int endX = std::min(prev_hsv.cols - 1, center.x + radius);
      int startY = std::max(0, center.y - radius);
      int endY = std::min(prev_hsv.rows - 1, center.y + radius);
      int min_dist_l2 = 256;
      int min_dist = 0;
      for (int y = startY; y <= endY; ++y) {
        for (int x = startX; x <= endX; ++x) {
          if (mask_.at<uchar>(bbox.y + y, bbox.x + x) == 0 || mask.at<uchar>(j, i) == 0) {
            continue;
          }
          cv::Vec3b curr_color = curr_hsv.at<cv::Vec3b>(y, x);
          if (curr_color[1] > 43) {
            continue;
          }
          if (abs(curr_color[2] - prev_color[2]) < min_dist_l2) {
            min_dist_l2 = abs(curr_color[2] - prev_color[2]);
            min_dist = curr_color[2] - prev_color[2];
          }
        }
      }
      if (min_dist_l2 > shadow_v_) {
        continue;
      }
      // std::cout << center << prev_color << "--" << min_point << curr_hsv.at<cv::Vec3b>(min_point)<<
      // std::endl;
      if (min_dist > 0) {
        light_pixs++;
        result_mask.at<uchar>(j, i) = 255;
      } else if (min_dist < 0) {
        dark_pixs++;
        result_mask.at<uchar>(j, i) = 0;
      }
    }
  }
  double score = fabs(light_pixs - dark_pixs) / double(cv::countNonZero(mask));
  imshow("light " + idx, mask);
  return score;
}

// 判断点是否在椭圆内部
bool isPointInsideEllipse(cv::Point2f point, cv::Point2f center, double a, double b) {
  // 点的坐标
  double x = point.x;
  double y = point.y;

  // 椭圆中心的坐标
  double x_c = center.x;
  double y_c = center.y;

  // 计算椭圆方程左边的值
  double ellipseValue = (std::pow(x - x_c, 2) / std::pow(a, 2)) + (std::pow(y - y_c, 2) / std::pow(b, 2));

  // 如果计算结果小于等于1，说明点在椭圆内部或边界上
  return ellipseValue <= 1.0;
}
// 计算点到椭圆边缘的距离
double pointToEllipseDistance(cv::Point pt, cv::RotatedRect ellipse) {
  cv::Point center = ellipse.center;
  float a = ellipse.size.width / 2.0f;   // 椭圆长轴半径
  float b = ellipse.size.height / 2.0f;  // 椭圆短轴半径

  // 将点坐标变换到以椭圆中心为原点的坐标系
  float x = pt.x - center.x;
  float y = pt.y - center.y;

  // 椭圆的标准方程: (x^2 / a^2) + (y^2 / b^2) = 1
  // 距离计算公式
  return sqrt((x * x) / (a * a) + (y * y) / (b * b));
}
// 计算直线与椭圆的交点
double computeEllipseIntersections(cv::Point point, cv::Point center, double a, double b) {
  // 点的坐标
  double x_p = point.x;
  double y_p = point.y;

  // 椭圆中心的坐标
  double x_c = center.x;
  double y_c = center.y;

  // 计算 x 和 y 方向的差值
  double dx = x_p - x_c;
  double dy = y_p - y_c;

  // 计算参数 t1 和 t2
  double denom = (dx * dx) / (a * a) + (dy * dy) / (b * b);
  double t1 = std::sqrt(1.0 / denom);
  double t2 = -t1;

  // 计算两个交点
  double x_intersect1 = x_c + t1 * dx;
  double y_intersect1 = y_c + t1 * dy;

  double x_intersect2 = x_c + t2 * dx;
  double y_intersect2 = y_c + t2 * dy;

  cv::Point intersection1(x_intersect1, y_intersect1);
  cv::Point intersection2(x_intersect2, y_intersect2);
  double dist1 = cv::norm(intersection1 - point);
  double dist2 = cv::norm(intersection2 - point);
  double dist = 0;
  if (dist1 < dist2) {
    dist = dist1;
  } else {
    dist = dist2;
  }
  if (!isPointInsideEllipse(point, center, a, b)) {
    dist = -dist;
  }
  return dist;
}

// 计算点集的加权分数
double calculateScore(const std::vector<cv::Point>& points, cv::RotatedRect ellipse) {
  double totalScore = 0.0;
  cv::Point center = ellipse.center;

  // 椭圆长短轴半径
  float a = ellipse.size.width / 2.0f;   // 长轴
  float b = ellipse.size.height / 2.0f;  // 短轴

  // 计算每个点到椭圆中心的距离和边缘的距离
  for (const auto& pt : points) {
    // double distToEdge = pointToEllipseDistance(pt, ellipse);
    double distToEdge = computeEllipseIntersections(pt, ellipse.center, a, b);
    double distToCenter = norm(pt - center);  // 点到中心的距离

    // 计算权重 (1 - distToEdge)
    // double weight = 1.0 - (distToCenter / sqrt(a * a + b * b));
    double weight = 1.0 - (distToCenter / (distToCenter + distToEdge));
    // 将点乘以权重
    totalScore += weight;
  }

  return totalScore / points.size();
}

float AntClpAlg::ContourSimilarity(const cv::Mat& prev_img, const cv::Mat& curr_img, std::string idx,
                                   cv::Rect boundingBox) {
  cv::Mat prev_canny, curr_canny;
  cv::Canny(prev_img, prev_canny, contour_low_thresh_, contour_high_thresh_);
  cv::Canny(curr_img, curr_canny, contour_low_thresh_, contour_high_thresh_);
  if (!cv::countNonZero(prev_canny) && !cv::countNonZero(curr_canny)) {
    return 0;
  }

  int radius = (contour_kernel_size_[mode_] - 1) / 2;
  cv::Mat prev_canny_diff = prev_canny.clone();
  cv::Mat curr_canny_diff = curr_canny.clone();
  for (size_t i = 0; i < prev_canny.cols; i++) {
    for (size_t j = 0; j < prev_canny.rows; j++) {
      cv::Point center(i, j);
      uchar prev_color = prev_canny.at<uchar>(j, i);
      if (prev_color == 0) {
        continue;
      }
      int startX = std::max(0, center.x - radius);
      int endX = std::min(prev_canny.cols - 1, center.x + radius);
      int startY = std::max(0, center.y - radius);
      int endY = std::min(prev_canny.rows - 1, center.y + radius);
      bool is_find = false;
      for (int y = startY; y <= endY; ++y) {
        for (int x = startX; x <= endX; ++x) {
          uchar curr_color = curr_canny.at<uchar>(y, x);
          if (prev_color == curr_color) {
            if (curr_canny_diff.at<uchar>(y, x) != 0) {
              curr_canny_diff.at<uchar>(y, x) = 0;
            }
            is_find = true;
          }
        }
      }
      if (is_find) {
        prev_canny_diff.at<uchar>(j, i) = 0;
      }
    }
  }
  cv::Mat canny_diff = prev_canny_diff + curr_canny_diff;
  // cv::Rect rect = boundingBox;
  // rect.x = MIN(radius, canny_diff.cols);
  // rect.y = MIN(radius, canny_diff.rows);
  // rect.width = MAX(0, rect.width - 2 * radius);
  // rect.height = MAX(0, rect.height - 2 * radius);
  // cv::Mat canny_diff2 = canny_diff(rect);
  // cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(morph_size[threq],
  // morph_size[threq])); cv::dilate(canny_diff2, canny_diff2, element, cv::Point(-1, -1));
  cv::imshow("img1 thre " + idx, canny_diff);
  // 筛选有效、无效边缘
  // 异物存在时，边缘差异图，面积与长度的比值（长度，分布）
  // std::vector<std::vector<cv::Point>> contours;
  // cv::findContours(canny_diff2, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  // double area = 0;
  // for (size_t i = 0; i < contours.size(); i++) {
  //   allPoints.insert(allPoints.end(), contours[i].begin(), contours[i].end());
  //   area += cv::contourArea(contours[i]);
  // cv::RotatedRect minRect = cv::minAreaRect(contours[i]);
  // area += cv::minAreaRect(contours[i]).size.area();
  // }
  int pixelCount = 0;
  std::vector<cv::Point> allPoints;
  for (int y = 0; y < canny_diff.rows; ++y) {
    for (int x = 0; x < canny_diff.cols; ++x) {
      if (canny_diff.at<uchar>(y, x) == 255) {
        allPoints.push_back(cv::Point(x, y));
        pixelCount++;
      }
    }
  }
  double lentgh_score = pixelCount / float(boundingBox.area());

  cv::RotatedRect ellipse =
      cv::RotatedRect(cv::Point2f(boundingBox.width / 2.0f, boundingBox.height / 2.0f),
                      cv::Size2f(boundingBox.width, boundingBox.height), 0);
  double distrib_score = calculateScore(allPoints, ellipse);
  double score = (lentgh_score * distrib_score) * 30;
  // cv::Mat img(boundingBox.size(), CV_8UC3, cv::Scalar(255, 255, 255));
  // cv::ellipse(img, ellipse, cv::Scalar(0, 255, 0), 2);
  //  for (const auto& pt : allPoints)
  // {
  //     cv::circle(img, pt, 3, cv::Scalar(0, 0, 255), -1);
  // }
  // cv::imshow("Points and Ellipse" + idx, img);
  // cout << "点集的加权分数: " << score << endl;
  printf("idx:%s, length score:%f, weight score:%f, average:%f\n", idx.c_str(), lentgh_score, distrib_score,
         score);
  /////////////////////////////////////////////////
  // cv::Mat prev = prev_canny(rect);
  // cv::Mat curr = curr_canny(rect);
  std::vector<std::vector<cv::Point>> contours1, contours2;
  cv::findContours(prev_canny, contours1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  cv::findContours(curr_canny, contours2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  cv::Mat contourImg = prev_img.clone();
  cv::drawContours(contourImg, contours1, -1, cv::Scalar(0, 0, 255), 1);  // 红色绘制
  cv::Mat contourImg2 = curr_img.clone();
  cv::drawContours(contourImg2, contours2, -1, cv::Scalar(0, 0, 255), 1);  // 红色绘制
  cv::imshow("Contours1" + idx, contourImg);
  cv::imshow("Contours2" + idx, contourImg2);
  // 轮廓进行比较
  // std::vector<float> scores;
  // for (size_t i = 0; i < contours2.size(); i++) {
  //   if (contours2[i].size() < 3) {
  //     continue;
  //   }
  //   float score = 9999999;
  //   for (size_t j = 0; j < contours1.size(); j++) {
  //     double match_score = cv::matchShapes(contours2[i], contours1[j], cv::CONTOURS_MATCH_I1, 0);
  //     if (match_score < score) {
  //       score = match_score;
  //     }
  //   }
  //   scores.push_back(score);
  // }
  // cv::imshow("img1 thre Channel" + idx, canny_img1);
  // cv::imshow("img2 thre Channel" + idx, canny_img2);
  // for (size_t i = 0; i < scores.size(); i++) {
  //   std::cout << "轮廓相似性得分 (越小越相似): " << contours2[i].size() << ":" << scores[i] <<
  //   std::endl;
  // }
  // cv::waitKey(0);
  return score;
}
////////////////////////////////////////////////////////////////
float HOGSimilarity(const cv::Mat& img1, const cv::Mat& img2) {
  cv::HOGDescriptor hog;
  std::vector<float> descriptors1, descriptors2;
  // cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
  // cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);
  hog.compute(img1, descriptors1);
  hog.compute(img2, descriptors2);

  // 比较特征向量相似性 (L2范数)
  double similarity = cv::norm(descriptors1, descriptors2, cv::NORM_L2);
  std::cout << "HOG Similarity: " << similarity << std::endl;
}
//////////////////////////////////////////////////////////////////

void AntClpAlg::Similarity(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask) {
  // IsShow = false;
  // std::vector<std::vector<cv::Point>> contours;  // 轮廓
  // std::vector<cv::Vec4i> hierarchy;
  // cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
  // int idx = 0;
  // int width = img1.cols;
  // int height = img1.rows;
  // for (size_t i = 0; i < contours.size(); i++) {
  //   const std::vector<cv::Point>& contour = contours[i];
  //   if (contour.size() < 3) {
  //     continue;
  //   }
  //   int enlarge_pix = param_.enlarge_pixs[mode_];
  //   cv::Rect bbox = cv::boundingRect(contour);
  //   cv::Rect boundingBox;
  //   boundingBox.x = MAX(bbox.x - enlarge_pix, 0);
  //   boundingBox.y = MAX(bbox.y - enlarge_pix, 0);
  //   boundingBox.width = MIN(width - boundingBox.x, bbox.width + 2 * enlarge_pix);
  //   boundingBox.height = MIN(height - boundingBox.y, bbox.height + 2 * enlarge_pix);
  //   // 创建掩码图像，大小为提取的轮廓区域大小
  //   cv::Mat roi_mask = cv::Mat::zeros(mask.size(), CV_8UC1);

  //   // 在掩码上绘制轮廓区域，将轮廓区域设为255
  //   // std::vector<std::vector<cv::Point>> contoursTransformed(1);
  //   // for (const auto& point : contours[0]) {
  //   //   contoursTransformed[0].emplace_back(point - boundingBox.tl());  // 调整坐标
  //   // }
  //   cv::drawContours(roi_mask, contours, i, cv::Scalar(255), cv::FILLED);

  //   // 提取图像中轮廓的区域
  //   cv::Mat roi1 = img1(boundingBox);  // 从原图中截取ROI区域
  //   // cv::Mat extractedRegion1 = cv::Mat::zeros(boundingBox.size(), img1.type());
  //   // // 将ROI区域与掩码叠加
  //   // roi1.copyTo(extractedRegion1, mask);
  //   cv::Mat roi2 = img2(boundingBox);  // 从原图中截取ROI区域
  //   // cv::Mat extractedRegion2 = cv::Mat::zeros(boundingBox.size(), img2.type());
  //   // // 将ROI区域与掩码叠加
  //   // roi2.copyTo(extractedRegion2, mask);
  //   // cv::imshow("hog 1", extractedRegion1);
  //   // cv::imshow("hog 2", extractedRegion2);
  //   // cv::waitKey(0);
  //   // cv::Mat img1cp = img1.clone();
  //   // cv::Mat img2cp = img2.clone();
  //   // HOGSimilarity(img1cp, img2cp);
  //   cv::Mat high_response_roi = high_response_mask_(boundingBox);
  //   int high_response_area = cv::countNonZero(high_response_roi);
  //   printf("id:%d, high response:%f\n", idx, high_response_area / cv::contourArea(contour));
  //   if (high_response_area / cv::contourArea(contour) > hit_high_response_threshold_) {
  //     // printf("id:%d hit high response find contour, area:%f\n", idx, cv::contourArea(contour));
  //     if (cv::contourArea(contour) > object_area_threshold_) bboxes_.push_back(bbox);
  //     // continue;
  //   }
  //   cv::Mat shadow_mask;
  //   float score = 0;
  //   if (mode_ == 0) {
  //     score = ContourSimilarity(roi1, roi2, std::to_string(idx), boundingBox);
  //   }
  //   IsShow = true;
  //   if (score > contour_threshold_ || mode_ == 1) {
  //     double shadow_score = ShadowDetect(bbox, shadow_mask, std::to_string(idx));
  //     if (shadow_score > shadow_threshold_) {
  //       printf("id:%d, is shadow\n", idx);
  //     } else {
  //       if (cv::contourArea(contour) > object_area_threshold_) bboxes_.push_back(bbox);
  //       printf("id:%d, shadow:%f, area:%f\n", idx, shadow_score, cv::contourArea(contour));
  //     }
  //   } else {
  //     // IsShow = false;
  //     printf("-------------id:%d, not find contour\n", idx);
  //   }
  //   idx++;
  // }
}

std::vector<Object> AntClpAlg::DetectObjects() {
  std::vector<Object> objects;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(object_mask_, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE,
                   cv::Point(0, 0));
  int idx = 0;
  int width = prevPreProImg_.cols;
  int height = prevPreProImg_.rows;
  int padding = padding_size_[mode_];
  for (size_t i = 0; i < contours.size(); i++) {
    const std::vector<cv::Point>& contour = contours[i];
    if (contour.size() < 3) {
      continue;
    }
    cv::Rect bbox = cv::boundingRect(contour);
    cv::Mat mask = cv::Mat::zeros(object_mask_.size(), CV_8UC1);
    cv::drawContours(mask, contours, i, cv::Scalar(255), cv::FILLED);
    cv::Mat high_response_roi = cv::Mat::zeros(high_response_mask_.size(), CV_8UC1);
    high_response_mask_.copyTo(high_response_roi, mask);
    int high_response_area = cv::countNonZero(high_response_roi);
    float hit_high_response_score = high_response_area / cv::contourArea(contour) * 10;
    if (hit_high_response_score > hit_high_response_threshold_) {
      Object obj;
      obj.id = idx;
      obj.bbox = bbox;
      obj.contour = contour;
      obj.confidence = hit_high_response_score;
      objects.push_back(obj);
      idx++;
      continue;
    }
    float contour_score = 0;
    if (mode_ == 0) {
      // 1.6
      cv::Rect boundingBox;
      boundingBox.x = MAX(bbox.x - padding, 0);
      boundingBox.y = MAX(bbox.y - padding, 0);
      boundingBox.width = MIN(width - boundingBox.x, bbox.width + 2 * padding);
      boundingBox.height = MIN(height - boundingBox.y, bbox.height + 2 * padding);
      cv::Mat roi1 = prevPreProImg_(boundingBox);
      cv::Mat roi2 = currPreProImg_(boundingBox);
      contour_score = ContourSimilarity(roi1, roi2, std::to_string(idx), boundingBox);
    }
    if (contour_score > contour_threshold_ || mode_ == 1) {
      if (shadow_suppression_) {
        cv::Mat shadow_mask;
        cv::Mat roi_msk = mask(bbox);
        double shadow_score = ShadowDetect(bbox, roi_msk, shadow_mask, std::to_string(idx));
        if (shadow_score > shadow_threshold_) {
          continue;
        }
      }
      Object obj;
      obj.id = idx;
      obj.bbox = bbox;
      obj.contour = contour;
      obj.confidence = contour_score;
      objects.push_back(obj);
      idx++;
    }
  }
  return objects;
}

std::vector<Object> AntClpAlg::ObjectsProgress(const std::vector<Object>& objects) {
  std::vector<Object> result;
  for (auto it = objects.begin(); it < objects.end(); it++) {
    if (cv::contourArea(it->contour) > object_area_threshold_){
      result.push_back(*it);
    }
  }
  return result;
}

//////////////////////////////////////////////////////////////////
/**
 * @brief process 算法处理函数
 * @param resultImg 目标检测结果图像
 * @param userData 用户数据指针，检测目标数，默认返回int类型
 * */
int AntClpAlg::process(cv::Mat& resultImg, void* userData) {
  //
  int* result = NULL;
  if (userData) {
    result = (int*)userData;
    *result = 0;
  }

  // check
  if (prevPreProImg_.empty() || currPreProImg_.empty()) {
    std::cout << "Process: image is empty" << std::endl;
    return -1;
  }

  // check
  if (prevPreProImg_.size() != currPreProImg_.size()) {
    std::cout << "Process: unequal size" << std::endl;
    return -1;
  }

  // 前背景 图像分割 提取差异部分 简单分割
  // image_segmentation(prevPreProImg_, currPreProImg_, object_mask_);
  segmentation();
  DEBUG_SHOW("segmentation_mask", object_mask_);
  // 二值前景过滤  形态学处理
  morphphological_process(object_mask_);
  DEBUG_SHOW("maskImg0", object_mask_);
  // ContourDiff(prevGrayImg_, currGrayImg_);
  // HOGSimilarity(prevPreProImg_, currPreProImg_);
  std::vector<Object> objects = DetectObjects();
  // Similarity(prevPreProImg_, currPreProImg_, object_mask_);
  // SIFT
  //  cv::Mat mask = object_mask_.clone();
  //  cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
  //  cv::erode(object_mask_, object_mask_, element, cv::Point(-1, -1));
  //  DEBUG_SHOW("erode object_mask_", object_mask_);
  //  cv::Mat prev = prevSrcImg_.clone();
  //  cv::Mat curr = currSrcImg_.clone();
  //  cv::Mat prev_masked_img;
  //  cv::bitwise_and(prev, prev, prev_masked_img, mask);
  //  cv::Mat curr_masked_img;
  //  cv::bitwise_and(curr, curr, curr_masked_img, mask);
  //  DEBUG_SHOW("prev_masked_img", prev_masked_img);
  //  DEBUG_SHOW("curr_masked_img", curr_masked_img);
  //  SiftFeature(prev_masked_img, curr_masked_img);
  // 其他 阴影检测
  // advance_process(prevHsvImg_, currHsvImg_, object_mask_);
  // DEBUG_SHOW("maskImg1", object_mask_);
  // 二值前景过滤  形态学处理
  //  morphphological_process(object_mask_);
  //  DEBUG_SHOW("maskImg2", object_mask_);
  // 标记
  // std::vector<cv::Rect> vBoundRect;
  // find_objdect(object_mask_, vBoundRect);

  // std::vector<int> picked;
  // box_filter(bboxes_, picked, NMS_THRESH);

  // 画矩形框
  currSrcImg_.copyTo(resultImg);
  for (auto it = objects.begin(); it < objects.end(); it++) {
    cv::rectangle(resultImg, it->bbox, cv::Scalar(0, 0, 255), 2);
  }
  if (objects.size() > 0) {
    IsShow = true;
  } else {
    IsShow = false;
  }
  // cv::cvtColor(object_mask_, object_mask_, cv::COLOR_GRAY2BGR);
  // if (!bboxes_.empty()) {
  //   for (auto it = picked.begin(); it < picked.end(); it++) {
  //     cv::rectangle(resultImg, bboxes_[*it], cv::Scalar(0, 0, 255), 2);
  //   }
  // }
  // bboxes_.clear();
  // DEBUG_SHOW("nms_object", object_mask_);
  // for (auto it = vBoundRect.begin(); it < vBoundRect.end(); it++) {
  //   cv::rectangle(object_mask_, *it, cv::Scalar(0, 0, 255), 2);
  // }
  DEBUG_SHOW("object", resultImg);
  // 返回检测结果
  if (result) {
    *result = objects.size();
  }

  // 释放图像资源 TODO
  // if (IsShow) {
  // cv::waitKey(0);
  // }
  return 0;
}

#if 0
/**
 * @brief image_segmentation 前背景分割函数  
 * @param img1 输入图像1 灰度图像
 * @param img2 输入图像2 灰度图像
 * @param maskImg  二值化图像
 * @return None
 * otsu 鲁棒性
 * */
void AntClpAlg::image_segmentation(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& maskImg)
{
	//check
	if(image1.empty() || image2.empty())
	{
		std::cout << "ImageSegmentation: input image can not be empty" << std::endl;
		return;
	}
	//
	if(image1.size() != image2.size())
	{
		std::cout << "ImageSegmentation: unequal size" << std::endl;
		return;
	}

	cv::Mat diffImg;
	cv::absdiff(image1, image2, diffImg);
    //滤波...？
	//
	cv::threshold(diffImg, maskImg, 0, 255, CV_THRESH_OTSU);

}

#else
/**
 * @brief image_segmentation 前背景分割函数   八邻域分割  利用局域相似性
 * @param img1 输入图像1 灰度图像
 * @param img2 输入图像2 灰度图像
 * @param maskImg  输出图像 二值化图像
 * @return None
 * */
void AntClpAlg::image_segmentation(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& maskImg) {
  // 局部-》全局
  cv::Mat grayImage1 = img1;
  cv::Mat grayImage2 = img2;
  // 比较数据与模型差异
  int rows = grayImage1.rows;
  int cols = grayImage1.cols;
  //
  // if(maskImg.empty())
  // {
  maskImg.create(rows, cols, CV_8UC1);
  // }
  // 设置全零
  maskImg.setTo(cv::Scalar::all(0));

  // 遍历图像
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      int matchCnt = 0;
      for (int sampleIndex = 0; sampleIndex < 9; sampleIndex++) {
        //
        int neiborRow = row + yOffset[sampleIndex];
        int neiborCol = col + xOffset[sampleIndex];
        // 边界限制
        if (neiborRow < 0) {
          neiborRow = 0;
        } else if (neiborRow >= rows) {
          neiborRow = rows - 1;
        }
        if (neiborCol < 0) {
          neiborCol = 0;
        } else if (neiborCol >= cols) {
          neiborCol = cols - 1;
        }

        int dis = abs(grayImage2.at<uchar>(row, col) - grayImage1.at<uchar>(neiborRow, neiborCol));
        if (dis < DEFAULT_COLOR_DISTANCE) {
          matchCnt++;
        }
        if (matchCnt > DEFAULT_MATCH_NUM) {
          break;
        }
      }
      // 背景
      if (matchCnt >= DEFAULT_MATCH_NUM) {
        continue;
      }
      // 前景
      maskImg.at<uchar>(row, col) = 255;

    }  // end for col
  }  // end for row
}

#endif

/**
 * @brief gradient_exaction  边缘检测
 * @param inImg 输入图像 灰度图像
 * @param inImg 输出图像 灰度图像
 * @return None
 * */
void AntClpAlg::gradient_exaction(const cv::Mat& inImg, cv::Mat& outImg) {
  // TODO
}

#if 0

/**
 * @brief calc_variance 计算邻域方差 ？？？
 * @param inImg 输入图像 灰度图像
 * @return None
 * */

void calc_variance(cv::Mat& inImg)
{

	int rows = inImg.rows;
	int cols = inImg.cols;
	for(int row = 0; row < rows; row++)
	{
		for(int col = 0; col < cols; col++)
		{
			if(!inImg.at<uchar>(row, col))
			{
				continue;	
			}

			float mean = 0.0;
			float variance = 0.0;
			uchar sample[9] = {0};
			for(int sampleIndex = 0; sampleIndex < 9; sampleIndex++)	
			{
				//
				int neiborRow = row + yOffset[sampleIndex];
				int neiborCol = col + xOffset[sampleIndex];
				//边界限制
				if(neiborRow < 0)
				{
					neiborRow = 0;
				}
				else if(neiborRow >= rows)
				{
					neiborRow = rows -1 ;
				}
				if(neiborCol < 0)
				{
					neiborCol = 0;
				}
				else if(neiborCol >= cols)
				{
					neiborCol = cols - 1;
				}

				sample[sampleIndex] = inImg.at<uchar>(neiborRow, neiborCol);
			}


			//calculate mean
			for(int i = 0; i < 9; i++)
			{
				mean += sample[i];
			}
			mean /= 9;

			for(int i = 0; i < 9; i++)
			{
				variance += pow((sample[i] - mean), 2);
			}
			variance /= (9 -1);

		}
	}
}
#endif

/**
 * @brief morphphological_process 形态学处理
 * @param maskImg 输入图像 二值化
 * @return None
 * */
void AntClpAlg::morphphological_process(cv::Mat& maskImg) {
  if (maskImg.empty()) {
    // std::cout << "Morphphological process: invalid input" << std::endl;
    return;
  } else {
    // std::cout << "maskImg type:" << maskImg.type() << std::endl;
  }

  // 腐蚀与膨胀 TODO
  cv::Mat element1 =
      getStructuringElement(cv::MORPH_RECT, cv::Size(morph_kernel_size_[mode_], morph_kernel_size_[mode_]));
  // cv::Mat element2 = getStructuringElement(
  //     cv::MORPH_RECT, cv::Size(morph_kernel_size_[mode_] + segment_kernel_size_[mode_],
  //     morph_kernel_size_[mode_] + segment_kernel_size_[mode_]));
  cv::erode(maskImg, maskImg, element1, cv::Point(-1, -1));
  cv::dilate(maskImg, maskImg, element1, cv::Point(-1, -1));
  std::vector<std::vector<cv::Point>> contours;  // 轮廓
  std::vector<cv::Vec4i> hierarchy;

  // cv::RETR_TREE检测所有轮廓
  // cv::CHAIN_APPROX_NONE 边界所有连续点
  cv::findContours(maskImg, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
  if (!contours.size()) {
    IsShow = false;
    return;
  } else {
    IsShow = true;
  }

  // 空洞填充
  for (int contIndex = 0; contIndex < contours.size(); contIndex++) {
    double contArea = cv::contourArea(contours[contIndex]);
    double perimeter = cv::arcLength(contours[contIndex], false);

    int firstLevel = -1, secondLevel = -1;
    firstLevel = hierarchy[contIndex][3];
    if (firstLevel >= 0) {
      secondLevel = hierarchy[firstLevel][3];
      if (secondLevel == -1) {
        if (contArea <= object_area_threshold_) {
          cv::drawContours(maskImg, contours, contIndex, cv::Scalar(255));
        }
      }
    }

    else if (firstLevel == -1) {
      // 过滤面积过小的轮廓 TODO
      if (contArea <= object_area_threshold_ || perimeter <= object_area_threshold_) {
        cv::drawContours(maskImg, contours, contIndex, cv::Scalar(0));
      }  // end ContArea
    }  //
  }
}

/**
 * @brief sortFunc 自定义contour 排序函数
 * @param a 点集
 * @param b  点集
 * @return None
 * */
// 从大到小
bool sortFunc(std::vector<cv::Point> a, std::vector<cv::Point> b) {
  return cv::contourArea(a) > cv::contourArea(b);
}

/**
 * @brief find_objdect 形态学处理及连通域标注
 * @param inImg 输入图像 二值化
 * @param vBoundRect 外接矩形
 * @return None
 * */
void AntClpAlg::find_objdect(const cv::Mat& inImg, std::vector<cv::Rect>& vBoundRect) {
  if (inImg.empty()) {
    std::cout << "Find object invalid input" << std::endl;
    return;
  }

  std::vector<std::vector<cv::Point>> contours;  // 轮廓
  std::vector<cv::Vec4i> hierarchy;

  // cv::RETR_EXTERNALj仅仅检测外部轮廓
  // cv::CHAIN_APPROX_NONE 边界所有连续点
  cv::findContours(inImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
  if (!contours.size()) {
    return;
  }

  // sort 从大到小 标准库函数 自定义排序函数
  std::sort(contours.begin(), contours.end(), sortFunc);

  //
  for (int contIndex = 0; contIndex < contours.size(); contIndex++) {
    double contArea = cv::contourArea(contours[contIndex]);
    // TODO
    if (contArea > object_area_threshold_)  //
    {
      // 最小外接矩形
      cv::Rect boundRect = cv::boundingRect(contours[contIndex]);
      vBoundRect.push_back(boundRect);
      // box_filter(boundRect, vBoundRect);
    }  //

  }  // end for each cont
}

/**
 * @brief box_filter 约束矩形过滤  整理
 * @param vBoundRect 矩形集合
 * @param picked 选择集合
 * @param nms_threshold 参考yolox 方法
 * @return None
 * 轮廓包含轮廓问题 轮廓交集处理
 * 暂时考虑 包含与被包含
 * (1) A包含B
 * (2) B包含A
 * (3) A与B相交 A > B
 * (4) A与B相交 A < B
 * */
void AntClpAlg::box_filter(std::vector<cv::Rect>& vBoundRect, std::vector<int>& picked,
                           float nms_threshold) {
  picked.clear();

  const int n = vBoundRect.size();

  std::vector<int> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = vBoundRect[i].area();
  }

  for (int i = 0; i < n; i++) {
    const cv::Rect& a = vBoundRect[i];

    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const cv::Rect& b = vBoundRect[picked[j]];

      // 是否包含
      if (b.x <= a.x && b.y <= a.y && b.width > a.width && b.height > a.height) {
        keep = 0;
        break;
      }

      // 交集
      //  intersection over union
      cv::Rect c = a & b;
      float inter_area = c.area();
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold) keep = 0;
    }
    if (keep) picked.push_back(i);
  }
}

#if 1
/**
 * @brief advance_process  光斑，阴影检测
 * @param img1 输入图像1 hsv
 * @param img2 输入图像2 hsv
 * @param maskImg 被修改图像 二值
 * @return None
 * */
void AntClpAlg::advance_process(cv::Mat& img1, cv::Mat& img2, cv::Mat& maskImg) {
  int rows = maskImg.rows;
  int cols = maskImg.cols;

  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      // 背景区域不处理
      if (!maskImg.at<uchar>(row, col)) {
        continue;
      }

      // 亮度比较
      int prevH = img1.at<cv::Vec3b>(row, col)[0];
      int prevS = img1.at<cv::Vec3b>(row, col)[1];
      int prevV = img1.at<cv::Vec3b>(row, col)[2];

      int currH = img2.at<cv::Vec3b>(row, col)[0];
      int currS = img2.at<cv::Vec3b>(row, col)[1];
      int currV = img2.at<cv::Vec3b>(row, col)[2];

      cv::Vec3b prevColorv = img1.at<cv::Vec3b>(row, col);
      cv::Vec3i prevColor = img1.at<cv::Vec3b>(row, col);
      prevColor[0] *= 0.5;
      prevColor[1] *= 0.3;
      prevColor[2] *= 0.2;

      cv::Vec3i currentColor = img2.at<cv::Vec3b>(row, col);
      currentColor[0] *= 0.5;
      currentColor[1] *= 0.3;
      currentColor[2] *= 0.2;

      int hmin = std::min(abs(prevH - currH), 180 - abs(prevH - currH));
      // int hmin = abs(prevH - currH);
      int smin = abs(prevS - currS);
      int vmin = abs(prevV - currV);

      // 一些特殊处理 ?
      // TODO
      // 若当前像素亮度高于上一次 判断为阴影
      if (1) {
        if (hmin <= DEFAULT_HUE_THVAL && smin <= DEFAULT_SATURATION_THVAL && vmin <= DEFAULT_VALUE_THVAL) {
          // 色度 "红", "橙", "黄", "绿", "青", "蓝", "紫"
          utils::hsvRegionGrowing(img2, img1, maskImg, col, row, currentColor, 3, 20, 0);
          maskImg.at<uchar>(row, col) = 0;
          std::cout << col << ", " << row << ", " << currentColor << std::endl;
          std::cout << std::endl;
          // DEBUG_SHOW("colorRegionGrowing", maskImg);
          // cv::waitKey(0);
        }
      } else {
        if ((prevS >= 43 && currS >= 43) && (currV >= 27 && prevV > 27) && hmin <= DEFAULT_HUE_THVAL &&
            smin <= DEFAULT_SATURATION_THVAL && vmin <= DEFAULT_VALUE_THVAL) {
          // 色度 "红", "橙", "黄", "绿", "青", "蓝", "紫"
          utils::hsvRegionGrowing(img2, img1, maskImg, col, row, currentColor, 5, 30, 0);
          maskImg.at<uchar>(row, col) = 0;
          std::cout << col << ", " << row << ", " << currentColor << std::endl;
          std::cout << std::endl;
          // DEBUG_SHOW("colorRegionGrowing", maskImg);
          // cv::waitKey(0);
        } else if ((currV <= 27 && prevV <= 27)) {
          // 黑
          maskImg.at<uchar>(row, col) = 0;
        } else if ((currV > 221 && prevV > 221) && (prevS < 43 && currS < 43)) {
          // 过白灰
          maskImg.at<uchar>(row, col) = 0;
          // 					std::cout << std::endl;
          // DEBUG_SHOW("colorRegionGrowing", maskImg);
          // cv::waitKey(0);
        } else {
          // 无解，无法适应所有色域，与背景或阴影相似的物体，--wq
        }
        // if ((currV > 221 && prevV > 221)) {
        //   // 过白
        // 	hsvRegionGrowing(img2, img1, maskImg, col, row, currentColor, 5,
        // 30, 0);
        //   maskImg.at<uchar>(row, col) = 0;
        // }
      }

    }  // end for each cols

  }  // end for each rows
}

#else
/**
 * @brief advance_process  光斑，阴影检测
 * @param img1 输入图像1 hsv
 * @param img2 输入图像2 hsv
 * @param maskImg 被修改图像 二值
 * @return None
 * */
void AntClpAlg::advance_process(cv::Mat& img1, cv::Mat& img2, cv::Mat& maskImg) {
  cv::Mat diff, hsvmask;
  cv::absdiff(img1, img2, diff);
  cv::blur(diff, diff, cv::Size(7, 7));
  inRange(diff, cv::Scalar(DEFAULT_HUE_THVAL, DEFAULT_SATURATION_THVAL, DEFAULT_VALUE_THVAL),
          cv::Scalar(180, 255, 255CHAIN_APPROX_NONEhsvImg2[3];
          cv::Mat h, s, v, hmask, smask, vmask;
          cv::Mat result;
          cv::split(img1, hsvImg1);
          cv::split(img2, hsvImg2);

          cv::absdiff(hsvImg1[0], hsvImg2[0], h);
          cv::absdiff(hsvImg1[1], hsvImg2[1], s);
          cv::absdiff(hsvImg1[2], hsvImg2[2], v);

          cv::threshold(h, hmask, 0, 255, CV_THRESH_OTSU);
          cv::threshold(s, smask, 0, 255, CV_THRESH_OTSU);
          cv::threshold(v, vmask, 0, 255, CV_THRESH_OTSU);

          //cv::add(hmask, smask, result);

          cv::imshow("hmask", hmask);
          cv::imshow("smask", smask);
          cv::imshow("vmask", vmask);


  */
  /*
          int vmin = 10, vmax = 256, smin = 30, hmin = 30;
          cv::namedWindow( "CamShift Demo", 0 );
          //setMouseCallback( "CamShift Demo", onMouse, 0 );
          cv::createTrackbar( "Vmin", "CamShift Demo", &vmin, 256, 0 );
          cv::createTrackbar( "Vmax", "CamShift Demo", &vmax, 256, 0 );
          cv::createTrackbar( "Smin", "CamShift Demo", &smin, 256, 0 );
          cv::createTrackbar( "Hmin", "CamShift Demo", &hmin, 256, 0 );

          cv::Mat diff, result;
          cv::absdiff(img1, img2, diff);

          for(;;)
          {
          inRange(diff, cv::Scalar(hmin, smin, MIN(vmin,vmax)),
                          cv::Scalar(180, 256, MAX(vmin, vmax)), result);

          cv::imshow("CamShift Demo", result);
          if(cv::waitKey(25) == 'q')
              break;
          }
  */
}

#endif

void onMouse(int event, int x, int y, int flags, void* param) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(param);
  switch (event) {
    case 1:  // 鼠标左键按下响应：返回坐标和灰度
      std::cout << "at(" << x << "," << y
                << ")value is:" << static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
      break;
    case 2:  // 鼠标右键按下响应：输入坐标并返回该坐标的灰度
      std::cout << "input(x,y)" << std::endl;
      std::cout << "x =" << std::endl;
      std::cin >> x;
      std::cout << "y =" << std::endl;
      std::cin >> y;
      std::cout << "at(" << x << "," << y
                << ")value is:" << static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
      break;
  }
}

void AntClpAlg::align(const cv::Mat& src, const cv::Mat& dst, cv::Mat& out) {
  // align_by_gray(src, dst, out);
  align_by_orb(src, dst, out);
}

void AntClpAlg::align_by_gray(const cv::Mat& src, const cv::Mat& dst, cv::Mat& out) {
  // 将图像转换为浮点类型
  cv::Mat img1_gray, img2_gray, img1_float, img2_float;
  cv::cvtColor(src, img1_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(dst, img2_gray, cv::COLOR_BGR2GRAY);
  img1_gray.convertTo(img1_float, CV_32F);
  img2_gray.convertTo(img2_float, CV_32F);

  // 初始化仿射矩阵 (2x3)
  cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);

  // 设置迭代参数
  int number_of_iterations = 5000;
  double termination_eps = 1e-10;
  cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, number_of_iterations,
                            termination_eps);

  // 使用 ECC 算法计算变换矩阵
  double cc = cv::findTransformECC(img1_float, img2_float, warp_matrix, cv::MOTION_AFFINE, criteria);
  std::cout << "transform:" << warp_matrix << std::endl;
  // 应用仿射变换
  cv::Mat alignedImage;
  cv::warpAffine(src, out, warp_matrix, dst.size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
}

void AntClpAlg::align_by_orb(const cv::Mat& src, const cv::Mat& dst, cv::Mat& out) {
  // 1. 使用 ORB 检测器检测关键点和描述符
  cv::Ptr<cv::SIFT> orb = cv::SIFT::create();
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  cv::Mat src_gray, dst_gray;
  cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(dst, dst_gray, cv::COLOR_BGR2GRAY);
  orb->detectAndCompute(src_gray, cv::noArray(), keypoints1, descriptors1);
  orb->detectAndCompute(dst_gray, cv::noArray(), keypoints2, descriptors2);

  // 2. 使用 BFMatcher 进行特征匹配
  cv::BFMatcher matcher(cv::NORM_L2, true);
  std::vector<cv::DMatch> matches;
  matcher.match(descriptors1, descriptors2, matches);

  // 3. 应用距离限制，设置阈值
  double distanceThreshold = 28.0;  // 根据需要设置合适的距离阈值
  std::vector<cv::DMatch> goodMatches;
  std::vector<cv::KeyPoint> goodkeypoints1, goodkeypoints2;
  int i = 0;
  for (auto& match : matches) {
    if (match.distance < distanceThreshold) {
      goodkeypoints1.push_back(keypoints1[match.queryIdx]);
      goodkeypoints2.push_back(keypoints2[match.trainIdx]);
      match.queryIdx = i;
      match.trainIdx = i;
      goodMatches.push_back(match);
      i++;
    }
  }

  // 4. 计算配准矩阵
  // std::vector<cv::Point2f> points1, points2;
  // for (const auto& match : goodMatches) {
  //   points1.push_back(keypoints1[match.queryIdx].pt);
  //   points2.push_back(keypoints2[match.trainIdx].pt);
  // }

  // if (points1.size() <= 3 || points2.size() <= 3) {
  //   std::cerr << "有效匹配点不足，无法计算变换矩阵" << std::endl;
  //   return ;
  // }

  // cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);  // 透视变换矩阵

  // 5. 应用变换
  // cv::warpPerspective(src, out, H, dst.size());

  // 绘制匹配结果
  // cv::Mat img_matches;
  // cv::drawMatches(src, goodkeypoints1, dst, goodkeypoints2, goodMatches, img_matches,
  // cv::Scalar::all(-1),
  //                 cv::Scalar::all(-1), std::vector<char>(),
  //                 cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  // 显示结果
  // cv::imshow("Matches", img_matches);
  // cv::waitKey(0);
}
// 计算两个颜色向量的欧氏距离
double RGBDistance(const cv::Vec3b& color1, const cv::Vec3b& color2) {
  double l2 = sqrt((int(color1[0]) - int(color2[0])) * (int(color1[0]) - int(color2[0])) +
                   (int(color1[1]) - int(color2[1])) * (int(color1[1]) - int(color2[1])) +
                   (int(color1[2]) - int(color2[2])) * (int(color1[2]) - int(color2[2])));
  return l2;
}
// 不分辨黑白灰
double HSVDistance(const cv::Vec3b& color1, const cv::Vec3b& color2) {
  if ((color1[1] < 43 && color2[1] < 43) || (color1[2] < 46 && color2[2] < 46)) {
    return -1;
  }
  double h_dist = abs(color1[0] - color2[0]);
  if (h_dist > 90) {
    h_dist -= 180;
  }
  double sv_dist = sqrt((int(color1[1]) - int(color2[1])) * (int(color1[1]) - int(color2[1])) +
                        (int(color1[2]) - int(color2[2])) * (int(color1[2]) - int(color2[2])));
  double l2 = sqrt(h_dist * h_dist + sv_dist * sv_dist);
  return l2;
}
bool IsBuleLight(const cv::Vec3b& color1, const cv::Vec3b& color2, double b_threshold,
                 double g_r_threshold) {
  // B通道变化大，G、R通道变化小
  double l2 = sqrt((int(color1[1]) - int(color2[1])) * (int(color1[1]) - int(color2[1])) +
                   (int(color1[2]) - int(color2[2])) * (int(color1[2]) - int(color2[2])));
  if (l2 < g_r_threshold) {
    if (fabs(int(color1[0]) - int(color2[0])) < b_threshold) {
      return true;
    }
  }
  return false;
}

// 查找与中心像素颜色相似的像素
bool AntClpAlg::IsSimilarPixels(const cv::Point& center, int kernel_size, double low_threshold,
                                double high_threshold, double& dist) {
  bool is_find = false;
  cv::Vec3b prev_bgr = prevPreProImg_.at<cv::Vec3b>(center);
  cv::Vec3b prev_hsv = prevHsvImg_.at<cv::Vec3b>(center);
  // 限制范围防止越界
  int radius = (kernel_size - 1) / 2;
  int startX = std::max(0, center.x - radius);
  int endX = std::min(mask_.cols - 1, center.x + radius);
  int startY = std::max(0, center.y - radius);
  int endY = std::min(mask_.rows - 1, center.y + radius);

  int rgb_close_dist = 255;
  int hsv_close_dist = 255;
  for (int y = startY; y <= endY; ++y) {
    for (int x = startX; x <= endX; ++x) {
      cv::Vec3b curr_bgr = currPreProImg_.at<cv::Vec3b>(cv::Point(x, y));
      cv::Vec3b curr_hsv = currHsvImg_.at<cv::Vec3b>(cv::Point(x, y));
      double rgb_distance = RGBDistance(prev_bgr, curr_bgr);
      double hsv_distance = HSVDistance(prev_hsv, curr_hsv);
      if (rgb_distance < rgb_close_dist) {
        rgb_close_dist = rgb_distance;
      }
      if (hsv_distance < hsv_close_dist) {
        hsv_close_dist = hsv_distance;
      }
      // 如果距离小于阈值，则认为颜色相似
      if ((rgb_distance < low_threshold && hsv_distance < low_threshold)) {
        is_find = true;
        // break;
      }

      if (!is_find && blue_suppression_) {
        bool is_bule_light = IsBuleLight(prev_bgr, curr_bgr, blue_threshold1, blue_threshold2);
        if (is_bule_light) {
          is_find = true;
          // break;
        }
      }
    }
  }
  if ((rgb_close_dist > high_threshold && hsv_close_dist > high_threshold)) {
    dist = 255;
  } else if ((rgb_close_dist < low_threshold && hsv_close_dist < low_threshold)) {
    dist = 0;
  } else {
    dist = MIN(rgb_close_dist, hsv_close_dist);
  }
  return is_find;
}

void AntClpAlg::segmentation() {
  object_mask_.create(mask_.size(), CV_8UC1);
  object_mask_.setTo(cv::Scalar::all(0));
  low_response_mask_.create(mask_.size(), CV_8UC1);
  low_response_mask_.setTo(cv::Scalar::all(0));
  high_response_mask_.create(mask_.size(), CV_8UC1);
  high_response_mask_.setTo(cv::Scalar::all(0));
  similarity_mask_.create(mask_.size(), CV_32F);
  for (int row = 0; row < mask_.rows; row++) {
    for (int col = 0; col < mask_.cols; col++) {
      cv::Point pt(col, row);
      if (mask_.at<uchar>(pt) == 0) {
        continue;
      }
      double dist = 0;
      bool isSim =
          IsSimilarPixels(pt, segment_kernel_size_[mode_], segment_low_thresh_, segment_high_thresh_, dist);
      if (dist == 0) {
        low_response_mask_.at<uchar>(pt) = 255;
      } else if (dist == 255) {
        high_response_mask_.at<uchar>(pt) = 255;
      }
      similarity_mask_.at<float>(pt) = dist;
      if (!isSim) {
        object_mask_.at<uchar>(row, col) = 255;
      }
    }
  }
  cv::Mat l2_visualized;
  // cv::normalize(similarity_mask_, l2_visualized, 0, 255, cv::NORM_MINMAX);
  // l2_visualized.convertTo(l2_visualized, CV_8U);
  cv::imshow("low_response_mask_", low_response_mask_);
  cv::imshow("high_response_mask_", high_response_mask_);
  cv::imshow("similarity_mask_", similarity_mask_);
}
