#include <iostream>
#include <map>
#include <opencv2/xfeatures2d.hpp>
#include "ant_clp_alg.h"
#include "utils.h"


void mouse_callback(int event, int x, int y, int flags, void* userdata);
void rectifyLightness(cv::Mat& image);
std::string MatchFrameByTime(std::string std, std::vector<cv::String> paths);

int main(int argc, char** argv) {
  printf("hello world \n");

  AntClpAlg alg;

  std::string cam_name = "468";
  cv::Rect roi;
  if (cam_name == "357") {
    roi = cv::Rect(500, 200, 940, 55);  
  } else if (cam_name == "157") {
    roi = cv::Rect(470, 230, 920, 55);  
  } else if (cam_name == "413") {
    roi = cv::Rect(370, 226, 940, 55);  
  } else if (cam_name == "424") {
    roi = cv::Rect(550, 220, 940, 55);  
  } else if (cam_name == "468") {
    roi = cv::Rect(720, 180, 920, 55);  
  }
  std::vector<cv::String> paths;
  cv::glob("/home/crrcdt123/datasets2/二门防夹数据/车间/cam" + cam_name + "-Start*.jpg",
           paths);
  std::vector<cv::String> end_paths;
  cv::glob("/home/crrcdt123/datasets2/二门防夹数据/车间/cam" + cam_name + "-End*.jpg",
           end_paths);
  // auto bg = cv::createBackgroundSubtractorMOG2();
  // auto knn = cv::createBackgroundSubtractorKNN();
  // for (size_t i = 0; i < paths.size(); i++) {
  //     std::cout << "idx:" << i << std::endl;
  //     std::string path = paths[i];
  //     cv::Mat img1 = cv::imread(path);
  //     if (img1.empty()) continue;
  //     img1 = img1(roi);
  //     cv::Mat mask = guss(bg, img1);
  //     // cv::Mat mask = GMG(knn, img1);
  //     cv::imshow("src", img1);
  //     cv::imshow("bg mask", mask);
  //     cv::waitKey(0);
  //     if (i == paths.size()-1) {
  //       paths = end_paths;
  //       i=0;
  //     }
  // }
  for (size_t i = 0; i < paths.size(); i++) {
    std::cout << "idx:" << i << std::endl;
    std::string path = paths[i];
    cv::Mat img1 = cv::imread(path);
    if (img1.empty()) continue;
    img1 = img1(roi);

    std::string end_path = MatchFrameByTime(path, end_paths);
    cv::Mat img2 = cv::imread(end_path);
    if (img2.empty()) continue;
    img2 = img2(roi);

    // save img
    std::string seq1 = "000" + std::to_string(i*2);
    seq1 = seq1.substr(seq1.length() - 3);
    cv::imwrite("/home/crrcdt123/git/PaddleSeg/contrib/QualityInspector/data/my_dataset/" + seq1 + ".jpg", img1);
    std::string seq2 = "000" + std::to_string(i*2+1);
    seq2 = seq2.substr(seq2.length() - 3);
    cv::imwrite("/home/crrcdt123/git/PaddleSeg/contrib/QualityInspector/data/my_dataset/" + seq2 + ".jpg", img2);
    continue;

    //
    cv::Mat img(cv::Size(img1.cols * 2, img1.rows), img1.type());
    cv::blur(img1, img1, cv::Size(3, 3));
    cv::blur(img2, img2, cv::Size(3, 3));
    img1.copyTo(img(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(img(cv::Rect(img1.cols, 0, img1.cols, img1.rows)));
    cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
    cv::imshow("src", img);
    cv::setMouseCallback("src", mouse_callback, reinterpret_cast<void*>(&img));


    //sift
    // cv::Mat sift_img1 = SiftFeature(img1, img2);

    // cv::Mat light = img1.clone();
    // rectifyLightness(light);
    // cv::imshow("light", light);

    cv::Mat img1_hsv, img2_hsv;
    cv::cvtColor(img1, img1_hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(img2, img2_hsv, cv::COLOR_BGR2HSV);

    /*
    统计颜色
    黑 灰 白 红 橙 黄 绿 青 蓝 紫
    颜色统计:idx:0 黑:758 灰:16908 白:649 红:161 橙:3858 黄:43 绿:5 青:0 蓝:1
    紫:0 颜色统计:idx:1 黑:659 灰:16756 白:272 红:47 橙:4517 黄:117 绿:2 青:0
    蓝:0 紫:0
    */
    int color_cnts[10] = {0};
    std::vector<std::map<int, std::vector<cv::Point>>> color_maps;
    std::vector<cv::Mat> imgs;
    imgs.emplace_back(img1_hsv);
    imgs.emplace_back(img2_hsv);
    for (int idx = 0; idx < imgs.size(); idx++) {
      std::map<int, std::vector<cv::Point>> color_map;
      for (int row = 0; row < img1.rows; row++) {
        for (int col = 0; col < img1.cols; col++) {
          uchar H = imgs[idx].at<cv::Vec3b>(row, col)[0];
          uchar S = imgs[idx].at<cv::Vec3b>(row, col)[1];
          uchar V = imgs[idx].at<cv::Vec3b>(row, col)[2];
          int color = classifyColor(H, S, V);
          color_map[color].emplace_back(cv::Point(col, row));
          color_cnts[color]++;
        }
      }
      printf(
          "颜色统计:idx:%d 黑:%d 灰:%d 白:%d 红:%d 橙:%d 黄:%d 绿:%d 青:%d "
          "蓝:%d "
          "紫:%d\n",
          idx, color_cnts[0], color_cnts[1], color_cnts[2], color_cnts[3],
          color_cnts[4], color_cnts[5], color_cnts[6], color_cnts[7],
          color_cnts[8], color_cnts[9]);
      color_maps.emplace_back(color_map);
      memset(&color_cnts, 0, 10 * sizeof(int));
    }
    printf(
        "颜色统计diff: 黑:%d 灰:%d 白:%d 红:%d 橙:%d 黄:%d 绿:%d 青:%d 蓝:%d "
        "紫:%d\n",
        color_maps[1][0].size() - color_maps[0][0].size(),
        color_maps[1][1].size() - color_maps[0][1].size(),
        color_maps[1][2].size() - color_maps[0][2].size(),
        color_maps[1][3].size() - color_maps[0][3].size(),
        color_maps[1][4].size() - color_maps[0][4].size(),
        color_maps[1][5].size() - color_maps[0][5].size(),
        color_maps[1][6].size() - color_maps[0][6].size(),
        color_maps[1][7].size() - color_maps[0][7].size(),
        color_maps[1][8].size() - color_maps[0][8].size(),
        color_maps[1][9].size() - color_maps[0][9].size());
    // draw show
    cv::Mat draw_img = img.clone();
    for (int i = 0; i < color_maps.size(); i++) {
      for (auto iter = color_maps[i].begin(); iter != color_maps[i].end();
           iter++) {
        int offsetX = i * img1.cols;
        for (auto pix : iter->second) {
          draw_img.at<cv::Vec3b>(pix.y, pix.x + offsetX)[0] =
              colors_[iter->first][0];
          draw_img.at<cv::Vec3b>(pix.y, pix.x + offsetX)[1] =
              colors_[iter->first][1];
          draw_img.at<cv::Vec3b>(pix.y, pix.x + offsetX)[2] =
              colors_[iter->first][2];
        }
      }
    }
    cv::imshow("draw_img", draw_img);

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

    alg.input_before(img1);
    alg.input_after(img2);

    int objNum = 0;
    cv::Mat mask;
    alg.process(mask, &objNum);
  }

  // cv::setMouseCallback("correct", mouse_callback,
  // reinterpret_cast<void*>(&correct_img));

  return 0;
}

//---------------------------------------------------//
// 鼠标事件 获取兴趣区域 start
// 鼠标事件回调
void mouse_callback(int event, int x, int y, int flags, void* userdata) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(userdata);  // 转换成Mat型指针
  cv::Mat src(im->size(), CV_8UC3);
  im->copyTo(src);  // 要复制到dst中，否则会直接改变im内容
  cv::Mat dst;
  cv::cvtColor(src, dst, cv::COLOR_BGR2HSV);
  cv::Mat correct_mat = dst(cv::Rect(0, 0, dst.cols / 2, dst.rows));
  switch (event) {
    case cv::EVENT_MOUSEMOVE:
      uchar preH, preS, preV, curH, curS, curV;
      int prevC, currC;
      if (x >= dst.cols / 2) {
        x -= dst.cols / 2;
      }
      preH = dst.at<cv::Vec3b>(y, x)[0];
      preS = dst.at<cv::Vec3b>(y, x)[1];
      preV = dst.at<cv::Vec3b>(y, x)[2];
      curH = dst.at<cv::Vec3b>(y, x + dst.cols / 2)[0];
      curS = dst.at<cv::Vec3b>(y, x + dst.cols / 2)[1];
      curV = dst.at<cv::Vec3b>(y, x + dst.cols / 2)[2];
      prevC = classifyColor(preH, preS, preV);
      currC = classifyColor(curH, curS, curV);
      printf(
          "pos:(%d, %d) preColor:%s (%u, %u, %u) curColor:%s "
          "(%u, %u, %u), ration: %f\n",
          x, y, colors_text_[prevC].c_str(), preH, preS, preV,
          colors_text_[currC].c_str(), curH, curS, curV, ((int)curS - (int)preS) / float((int)preV - (int)curV));

      cv::circle(src, cv::Point(x + dst.cols / 2, y), 2, cv::Scalar(0, 0, 255));
      cv::circle(src, cv::Point(x, y), 2, cv::Scalar(0, 0, 255));
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



std::string MatchFrameByTime(std::string std, std::vector<cv::String> paths) {
  int std_time = 0;
  size_t len = std.length();
  size_t found = std.find_last_of("-");
  if (found != std::string::npos) {
    std_time = std::stoi(std.substr(found + 1, len - found - 4));
  }

  int min_dist = 9999999;
  int min_idx = -1;
  for (size_t i = 0; i < paths.size(); i++) {
    std::string match = paths[i];
    int match_time = 0;
    size_t len = match.length();
    size_t found = match.find_last_of("-");
    if (found != std::string::npos) {
      match_time = std::stoi(match.substr(found + 1, len - found - 4));
      int dist = abs(std_time - match_time);
      if (dist < min_dist) {
        min_dist = dist;
        min_idx = i;
      }
    } else {
      printf("match fasled");
    }
  }
  printf("time dist:%d, path:%s \n", min_dist, paths[min_idx].c_str());
  if (min_dist < 100000)
    return paths[min_idx];
  else
    return "";
}