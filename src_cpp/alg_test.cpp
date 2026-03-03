#include <fstream>
#include <iostream>
#include <map>
// #include <opencv2/xfeatures2d.hpp>

#include "ant_clp_alg.h"
#include "utils.h"

void mouse_callback(int event, int x, int y, int flags, void* userdata);
void rectifyLightness(cv::Mat& image);
bool MatchFrameByTime(std::string& match_path, std::string std, std::vector<cv::String>& paths);
bool LoadLabels(std::string path, std::map<std::string, std::vector<std::vector<cv::Point>>>& labels);
bool isTest = 1;
double time_dist_ = -1.3e8;
  // 基于RGB(全色彩)和HSV(非黑灰白)双色彩模式、双阈值进行前后背景分割，提取色彩变化区域。变化满足高阈值直接输出，高阈值与低阈值之间的模糊范围启用边缘差异检测，小于低阈值放弃。
  // 受现场光线遮挡影响，
  // 大roi内色彩呈现些许变化，但纹理可能保持一致，人遮挡光线会造成画面昏暗，纹理弱化甚至消失（待改善），需要对色彩模糊范围启用边缘差异检测(有异物必然会存在边缘，且背景边缘单一，基于边缘辅助能够有效避免大面积色彩影响)
  // 小roi内色彩变化弱，且背景边缘复杂，不易与异物边缘区分。故不考虑光线的影响
  // 1.6
  // 大ROI一切正常，测试小ROI，当前数据有个异常。明天检查
int main(int argc, char** argv) {

  // cv::Mat showIMg = cv::Mat::zeros(cv::Size(960, 540), CV_8UC1);
  // cv::line(showIMg, cv::Point(307, 537), cv::Point(508, 537), cv::Scalar(255));
  // cv::line(showIMg, cv::Point(508, 537), cv::Point(479, 106), cv::Scalar(255));
  // cv::line(showIMg, cv::Point(479, 106), cv::Point(476, 106), cv::Scalar(255));
  // cv::line(showIMg, cv::Point(476, 106), cv::Point(307, 537), cv::Scalar(255));
  // cv::imshow("img", showIMg);
  // cv::waitKey(0);
  // return 1;
  AntClpAlg alg;
  std::map<std::string, std::vector<std::vector<cv::Point>>> labels;
  // 082810_delay_1s  1012_delay_1.5
  // std::string root = "/home/crrcdt123/二门数据/082810_delay_1s";
  // std::string label_path = "/home/crrcdt123/二门数据/images_labels/infos.txt";
  std::string root = "/home/crrcdt123/glam/crrc/datasets/s8/2door/anti_clamp_tradition-20241102-0828/TC1/";
  std::string label_path =
  "/media/crrcdt123/glam/crrc/datasets/s8/2door/20240923/081910/new2//infos.txt";
  std::vector<std::string> cams_name = {"cam113", "cam124", "cam157", "cam168", "cam213", "cam224",
                                        "cam257", "cam268", "cam313", "cam324", "cam357", "cam368",
                                        "cam413", "cam424", "cam457", "cam468", "cam513", "cam524",
                                        "cam557", "cam568", "cam613", "cam624", "cam657", "cam668"};
  LoadLabels(label_path, labels);
  int mode = 1;
  for (size_t cam_idx = 1; cam_idx < cams_name.size(); cam_idx++) {
    std::string cam_name = cams_name[cam_idx];
    // cam_name = "cam513";
    // const std::vector<cv::Point>& roi = labels[cam_name][mode];
    std::vector<cv::String> start_paths, end_paths;
    cv::glob(root + "/*Start*.jpg", start_paths);
    cv::glob(root + "/*Result*.jpg", end_paths);
    // auto bg = cv::createBackgroundSubtractorMOG2();
    // auto knn = cv::createBackgroundSubtractorKNN();
    // for (size_t i = 0; i < start_paths.size(); i++) {
    //     std::cout << "idx:" << i << std::endl;
    //     std::string path = start_paths[i];
    //     cv::Mat img1 = cv::imread(path);
    //     if (img1.empty()) continue;
    //     img1 = img1(roi);
    //     cv::Mat mask = guss(bg, img1);
    //     // cv::Mat mask = GMG(knn, img1);
    //     cv::imshow("src", img1);
    //     cv::imshow("bg mask", mask);
    //     cv::waitKey(0);
    //     if (i == start_paths.size()-1) {
    //       start_paths = end_paths;
    //       i=0;
    //     }
    // }
    for (size_t i = 0; i < end_paths.size(); i++) {
      std::cout << "idx:" << i << std::endl;
      std::string path = end_paths[i];
      if (path.find(cam_name) == std::string::npos) {
        continue;
      }

      std::string start_path;
      if (!MatchFrameByTime(start_path, path, start_paths)) {
        // continue;
        return 1;
      }
      cv::Mat image2 = cv::imread(path);
      if (image2.empty()) continue;
      cv::Mat mask = cv::Mat::zeros(image2.size(), CV_8UC1);
      // cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{roi}, cv::Scalar(255));
      // cv::Rect bounding_box = cv::boundingRect(roi);
      // cv::Mat img2 = cv::Mat::zeros(bounding_box.size(), image2.type());
      // cv::Mat polygon_region = image2(bounding_box);
      // polygon_region.copyTo(img2, mask(bounding_box));
      cv::Mat image1 = cv::imread(start_path);
      if (image1.empty()) continue;
      
      // cv::Mat img1 = cv::Mat::zeros(bounding_box.size(), image1.type());
      // polygon_region = image1(bounding_box);
      // polygon_region.copyTo(img1, mask(bounding_box));

      // std::cout << "img1:" << start_path << std::endl;
      // std::cout << "img2:" << path << std::endl;

      // alg.SetMask(mask(bounding_box), mode);
      // alg.input_before(img1);
      // alg.input_after(img2);

      // int objNum = 0;
      // cv::Mat mask1;
      // alg.process(mask1, &objNum);
      // cv::blur(image1, image1, cv::Size(5, 5));
      // cv::blur(image2, image2, cv::Size(5, 5));
      cv::resize(image1, image1, cv::Size(960, 640));
      cv::resize(image2, image2, cv::Size(960, 640));
      cv::Mat img(cv::Size(image1.cols * 2, image1.rows), image1.type());
      image1.copyTo(img(cv::Rect(0, 0, image1.cols, image1.rows)));
      image2.copyTo(img(cv::Rect(image1.cols, 0, image1.cols, image1.rows)));
      cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
      cv::imshow("img", img);
      // 等待按键
        char key = (char)cv::waitKey(0); // 等待按键，无时间限制

        // 根据按键进行操作
        if (key == 27) { // ESC键退出
            break;
        } else if (key == 81) { // 左方向键
            i--;
        } else if (key == 83) { // 右方向键
            i++;
        }
        i--;
      continue;
      // cv::Mat roi_img(cv::Size(img1.cols, img1.rows * 2), img1.type());
      // img1.copyTo(roi_img(cv::Rect(0, 0, img1.cols, img1.rows)));
      // img2.copyTo(roi_img(cv::Rect(0, img1.rows, img1.cols, img1.rows)));
      // // cv::blur(roi_img, roi_img, cv::Size(3, 3));
      // cv::namedWindow("roi_img", cv::WINDOW_AUTOSIZE);
      // cv::imshow("roi_img", roi_img);
      // cv::setMouseCallback("roi_img", mouse_callback, reinterpret_cast<void*>(&roi_img));
      // if (isTest) {
      //   cv::waitKey(0);
      // } else {
      //   if (alg.IsShow) {
      //     cv::waitKey(0);
      //   } else {
      //     // cv::waitKey(100);
      //   }
      // }
    }
    if (0) {
      // sift
      //  cv::Mat sift_img1 = SiftFeature(img1, img2);

      // cv::Mat light = img1.clone();
      // rectifyLightness(light);
      // cv::imshow("light", light);

      // cv::Mat img1_hsv, img2_hsv;
      // cv::cvtColor(img1, img1_hsv, cv::COLOR_BGR2HSV);
      // cv::cvtColor(img2, img2_hsv, cv::COLOR_BGR2HSV);

      /*
      统计颜色
      黑 灰 白 红 橙 黄 绿 青 蓝 紫
      颜色统计:idx:0 黑:758 灰:16908 白:649 红:161 橙:3858 黄:43 绿:5 青:0 蓝:1
      紫:0 颜色统计:idx:1 黑:659 灰:16756 白:272 红:47 橙:4517 黄:117 绿:2 青:0
      蓝:0 紫:0
      */
      // int color_cnts[10] = {0};
      // std::vector<std::map<int, std::vector<cv::Point>>> color_maps;
      // std::vector<cv::Mat> imgs;
      // imgs.emplace_back(img1_hsv);
      // imgs.emplace_back(img2_hsv);
      // for (int idx = 0; idx < imgs.size(); idx++) {
      //   std::map<int, std::vector<cv::Point>> color_map;
      //   for (int row = 0; row < img1.rows; row++) {
      //     for (int col = 0; col < img1.cols; col++) {
      //       uchar H = imgs[idx].at<cv::Vec3b>(row, col)[0];
      //       uchar S = imgs[idx].at<cv::Vec3b>(row, col)[1];
      //       uchar V = imgs[idx].at<cv::Vec3b>(row, col)[2];
      //       int color = classifyColor(H, S, V);
      //       color_map[color].emplace_back(cv::Point(col, row));
      //       color_cnts[color]++;
      //     }
      //   }
      //   printf(
      //       "颜色统计:idx:%d 黑:%d 灰:%d 白:%d 红:%d 橙:%d 黄:%d 绿:%d 青:%d "
      //       "蓝:%d "
      //       "紫:%d\n",
      //       idx, color_cnts[0], color_cnts[1], color_cnts[2], color_cnts[3],
      //       color_cnts[4], color_cnts[5], color_cnts[6], color_cnts[7],
      //       color_cnts[8], color_cnts[9]);
      //   color_maps.emplace_back(color_map);
      //   memset(&color_cnts, 0, 10 * sizeof(int));
      // }
      // printf(
      //     "颜色统计diff: 黑:%d 灰:%d 白:%d 红:%d 橙:%d 黄:%d 绿:%d 青:%d 蓝:%d "
      //     "紫:%d\n",
      //     color_maps[1][0].size() - color_maps[0][0].size(),
      //     color_maps[1][1].size() - color_maps[0][1].size(),
      //     color_maps[1][2].size() - color_maps[0][2].size(),
      //     color_maps[1][3].size() - color_maps[0][3].size(),
      //     color_maps[1][4].size() - color_maps[0][4].size(),
      //     color_maps[1][5].size() - color_maps[0][5].size(),
      //     color_maps[1][6].size() - color_maps[0][6].size(),
      //     color_maps[1][7].size() - color_maps[0][7].size(),
      //     color_maps[1][8].size() - color_maps[0][8].size(),
      //     color_maps[1][9].size() - color_maps[0][9].size());
      // // draw show
      // cv::Mat draw_img = img.clone();
      // for (int i = 0; i < color_maps.size(); i++) {
      //   for (auto iter = color_maps[i].begin(); iter != color_maps[i].end();
      //        iter++) {
      //     int offsetX = i * img1.cols;
      //     for (auto pix : iter->second) {
      //       draw_img.at<cv::Vec3b>(pix.y, pix.x + offsetX)[0] =
      //           colors_[iter->first][0];
      //       draw_img.at<cv::Vec3b>(pix.y, pix.x + offsetX)[1] =
      //           colors_[iter->first][1];
      //       draw_img.at<cv::Vec3b>(pix.y, pix.x + offsetX)[2] =
      //           colors_[iter->first][2];
      //     }
      //   }
      // }
      // cv::imshow("draw_img", draw_img);

      // 亮度统一
      // for (int row = 0; row < img1.rows; row++) {
      //   for (int col = 0; col < img1.cols; col++) {
      //     uchar preV = img1_hsv.at<cv::Vec3b>(row, col)[2];
      //     uchar curV = img2_hsv.at<cv::Vec3b>(row, col)[2];
      //     if (curV > preV && preV >= 43) {
      //       img1_hsv.at<cv::Vec3b>(row, col)[2] = curV;
      //     } else {
      //       img2_hsv.at<cv::Vec3b>(row, col)[2] = preV;
      //     }
      //   }
      // }
      // cv::Mat correct_img = img.clone();
      // cv::cvtColor(img1_hsv, img1, cv::COLOR_HSV2BGR);
      // cv::cvtColor(img2_hsv, img2, cv::COLOR_HSV2BGR);
      // img1.copyTo(correct_img(cv::Rect(0, 0, img1.cols, img1.rows)));
      // img2.copyTo(correct_img(cv::Rect(img1.cols, 0, img1.cols, img1.rows)));
      // cv::imshow("correct", correct_img);

      // // gray
      // cv::Mat gray_img;
      // cv::cvtColor(correct_img, gray_img, cv::COLOR_BGR2GRAY);
      // cv::imshow("gray_img", gray_img);
    }
  }
  return 0;
}

//---------------------------------------------------//
// 鼠标事件 获取兴趣区域 start
// 鼠标事件回调
void mouse_callback(int event, int x, int y, int flags, void* userdata) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(userdata);  // 转换成Mat型指针
  cv::Mat src(im->size(), CV_8UC3);
  im->copyTo(src);  // 要复制到dst中，否则会直接改变im内容
  cv::Mat dst = src;
  cv::Mat hsv_mat;
  int row = dst.rows / 2;
  cv::cvtColor(src, hsv_mat, cv::COLOR_BGR2HSV);
  // cv::Mat correct_mat = dst(cv::Rect(0, 0, dst.cols / 2, dst.rows));
  double l2;
  switch (event) {
    case cv::EVENT_MOUSEMOVE:
      uchar preH, preS, preV, curH, curS, curV, alignH, alignS, alignV;
      uchar preB, preG, preR, curB, curG, curR;
      int prevC, currC;
      y = y % row;
      preB = dst.at<cv::Vec3b>(y, x)[0];
      preG = dst.at<cv::Vec3b>(y, x)[1];
      preR = dst.at<cv::Vec3b>(y, x)[2];
      curB = dst.at<cv::Vec3b>(y + row, x)[0];
      curG = dst.at<cv::Vec3b>(y + row, x)[1];
      curR = dst.at<cv::Vec3b>(y + row, x)[2];
      preH = hsv_mat.at<cv::Vec3b>(y, x)[0];
      preS = hsv_mat.at<cv::Vec3b>(y, x)[1];
      preV = hsv_mat.at<cv::Vec3b>(y, x)[2];
      curH = hsv_mat.at<cv::Vec3b>(y + row, x)[0];
      curS = hsv_mat.at<cv::Vec3b>(y + row, x)[1];
      curV = hsv_mat.at<cv::Vec3b>(y + row, x)[2];
      // alignH = dst.at<cv::Vec3b>(y + 2*row, x)[0];
      // alignS = dst.at<cv::Vec3b>(y + 2*row, x)[1];
      // alignV = dst.at<cv::Vec3b>(y + 2*row, x)[2];
      // prevC = utils::classifyColor(preH, preS, preV);
      // currC = utils::classifyColor(curH, curS, curV);

      l2 = sqrt((int(preB) - int(curB)) * (int(preB) - int(curB)) +
                (int(preG) - int(curG)) * (int(preG) - int(curG)) +
                (int(preR) - int(curR)) * (int(preR) - int(curR)));
      // differ = cv::Mat(dst.at<cv::Vec3b>(y, x), CV_32FC3) - cv::Mat(dst.at<cv::Vec3b>(y, x + dst.cols /
      // 2), CV_32FC3);
      printf(
          "pos:(%d, %d) pre:(%u, %u, %u) cur: "
          "(%u, %u, %u), dist: %f, pre:(%u, %u, %u) cur:(%u, %u, %u)\n",
          x, y, preB, preG, preR, curB, curG, curR, l2, preH, preS, preV, curH, curS, curV);
      // printf(
      //     "pos:(%d, %d) preColor:%s (%u, %u, %u) curColor:%s "
      //     "(%u, %u, %u), ration: %f\n",
      //     x, y, utils::colors_text_[prevC].c_str(), preH, preS, preV,
      //     utils::colors_text_[currC].c_str(), curH, curS, curV, ((int)curS - (int)preS) / float((int)preV
      //     - (int)curV));

      cv::circle(src, cv::Point(x, y), 2, cv::Scalar(0, 0, 255));
      cv::circle(src, cv::Point(x, y + row), 2, cv::Scalar(0, 0, 255));
      cv::circle(src, cv::Point(x, y + 2 * row), 2, cv::Scalar(0, 0, 255));
      cv::imshow("mapping", src);
      cv::waitKey(1);
      break;
    default:
      break;
  }
}

void rectifyLightness(cv::Mat& image) {
  // 转换为Lab颜色空间
  cv::Mat lab_image;
  cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);

  // 分离通道
  std::vector<cv::Mat> lab_channels;
  cv::split(lab_image, lab_channels);
  cv::imshow("l", lab_channels[0]);

  // 对亮度通道进行直方图均衡化
  cv::equalizeHist(lab_channels[0], lab_channels[0]);

  // 合并处理后的通道
  cv::merge(lab_channels, lab_image);

  // 转换回BGR颜色空间
  cv::Mat result_image;
  cv::cvtColor(lab_image, result_image, cv::COLOR_Lab2BGR);
  image = result_image.clone();
}

// 不支持跨天匹配
bool MatchFrameByTime(std::string& match_path, std::string src_str, std::vector<cv::String>& paths) {
  int ori_data = 0;
  int ori_time = 0;
  std::string ori_name = "";
  size_t len = src_str.length();
  size_t found = src_str.find_last_of("-");
  if (found != std::string::npos) {
    ori_name = src_str.substr(found - 22, 6);
    ori_data = std::stoi(src_str.substr(found - 8, 8));
    ori_time = std::stoi(src_str.substr(found + 1, len - found - 4));
    int siz = src_str.substr(found + 1, len - found - 4).size();
    if (src_str.substr(found + 1, len - found - 4).size() == 9) {
      ori_time *= 10;
    }
  }

  int min_dist = -999999999;
  int min_idx = -1;
  for (size_t i = 0; i < paths.size(); i++) {
    std::string match = paths[i];
    int match_time = 0;
    size_t len = match.length();
    size_t found = match.find_last_of("-");
    if (i == 449) {
      int kk = 0;
    }
    if (found != std::string::npos) {
      std::string match_name = match.substr(found - 21, 6);
      int match_data = std::stoi(match.substr(found - 8, 8));
      if (match_data != ori_data || match_name != ori_name) {
        continue;
      }
      match_time = std::stoi(match.substr(found + 1, len - found - 4));
      if (match.substr(found + 1, len - found - 4).size() == 9) {
        match_time *= 10;
      }
      int dist = match_time - ori_time;
      if (dist < 0 && dist > min_dist) {
        min_dist = dist;
        min_idx = i;
      }
    }
  }
  if (min_idx >= 0 && min_dist > time_dist_) {
    printf("time dist:%d, path:%s \n", min_dist, paths[min_idx].c_str());
    match_path = paths[min_idx];
    return true;
  } else {
    printf("match fasled");
    return false;
  }
}

bool parsePoint(const std::string& input, cv::Point& pt) {
  // 去掉方括号
  size_t pos = input.find("[");
  if (pos == std::string::npos) {
    return false;
  }
  std::string cleanedInput = input.substr(pos + 1, input.size() - pos - 1);

  // 使用 stringstream 解析字符串
  std::stringstream ss(cleanedInput);
  std::string xStr, yStr;

  // 以逗号为分隔符读取 x 和 y
  std::getline(ss, xStr, ',');
  std::getline(ss, yStr, ',');

  // 转换为整数并返回 cv::Point
  pt.x = static_cast<float>(std::round(std::stod(xStr)));
  pt.y = static_cast<float>(std::round(std::stod(yStr)));

  return true;
}

bool LoadLabels(std::string path, std::map<std::string, std::vector<std::vector<cv::Point>>>& labels) {
  std::ifstream file(path);  // 打开文件
  if (!file.is_open()) {
    std::cerr << "Unable to open file" << std::endl;
    return false;  // 文件打开失败，返回错误
  }

  std::string line;
  std::string content;
  bool isNewCam = false;
  bool isNewRoi = false;
  std::string cam_name;
  std::vector<cv::Point> polgon;
  std::vector<std::vector<cv::Point>> polgons;
  // 逐行读取文件内容
  while (std::getline(file, line)) {
    if (line.find(".jpg") != std::string::npos) {
      size_t pos = line.find("-");
      cam_name = line.substr(0, pos);
      isNewCam = true;
    } else if (line.find("roi") != std::string::npos || line == "") {
      if (polgon.size() != 0) {
        labels[cam_name].push_back(polgon);
        polgon.clear();
      }
    } else {
      cv::Point pt;
      if (parsePoint(line, pt)) {
        polgon.push_back(pt);
      }
    }
  }

  file.close();  // 关闭文件
  return true;
}