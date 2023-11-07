
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include<iostream>
#include <chrono>
#include <thread>
#include "utils.h"

void openCamera();

void SiftFeature();

void printHSV();

void mouse_callback(int event, int x, int y, int flags, void* userdata);

int main()
{
  // printHSV();
  // openCamera();
  uchar a = '3';
  uchar b = '5';
  int c = (int)a * 0.1;
  uchar smin = abs(a - b);
  std::cout << smin << std::endl;
	return 0;
}

void openCamera() {
  cv::VideoCapture cap;
  cap.open(0);
  // cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(320));
  // cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(240));
  if (!cap.isOpened()) {
      throw std::runtime_error("Could not open video stream!");
  }
  auto bg = cv::createBackgroundSubtractorMOG2();
  auto knn = cv::createBackgroundSubtractorKNN();
  while (1)
  {
    cv::Mat img;
    cap.read(img);
    cv::Mat mask = guss(bg, img);
    // cv::Mat mask = GMG(knn, img);
    cv::imshow("bg mask", mask);
    cv::waitKey(30);
  }
}

void printHSV() {
  std::string path = "/home/crrcdt123/git/ros2_train_eyes/anti_clamp/data/图片1.png";
  cv::Mat img = cv::imread(path);
  /*
  统计颜色
  黑 灰 白 红 橙 黄 绿 青 蓝 紫
  颜色统计:idx:0 黑:758 灰:16908 白:649 红:161 橙:3858 黄:43 绿:5 青:0 蓝:1
  紫:0 颜色统计:idx:1 黑:659 灰:16756 白:272 红:47 橙:4517 黄:117 绿:2 青:0
  蓝:0 紫:0
  */
  cv::Mat im_hsv;
  cv::cvtColor(img, im_hsv, cv::COLOR_BGR2HSV);
  int color_cnts[10] = {0};
  std::vector<std::map<int, std::vector<cv::Point>>> color_maps;
  std::vector<cv::Mat> imgs;
  imgs.emplace_back(im_hsv);
  for (int idx = 0; idx < imgs.size(); idx++) {
    std::map<int, std::vector<cv::Point>> color_map;
    for (int row = 0; row < img.rows; row++) {
      for (int col = 0; col < img.cols; col++) {
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
  // draw show
  cv::Mat draw_img = img.clone();
  for (int i = 0; i < color_maps.size(); i++) {
    for (auto iter = color_maps[i].begin(); iter != color_maps[i].end();
          iter++) {
      int offsetX = i * img.cols;
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

  cv::Mat enhancedEdgesImage;
  cv::Laplacian(img, enhancedEdgesImage, CV_8U, 3, 1, 0, cv::BORDER_DEFAULT);
  // 对原始图像和边缘增强后的图像进行融合，突出边缘
  cv::Mat finalImage = img + enhancedEdgesImage;

  SiftFeature(img);


  cv::Mat hsvImage;
  cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);

  // 阴影检测
  cv::Mat shadowMask;
  cv::inRange(hsvImage, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 100), shadowMask);

  // 亮度校正
  cv::Mat correctedImage = img.clone();
  for (int y = 0; y < img.rows; y++) {
      for (int x = 0; x < img.cols; x++) {
          if (shadowMask.at<uchar>(y, x) > 0) {
              correctedImage.at<cv::Vec3b>(y, x) *= 1.5;  // 亮度加权
          }
      }
  }

  cv::imshow("Shadow Mask", shadowMask);
  cv::imshow("Corrected Image", correctedImage);
  cv::imshow("Enhanced Edges Image", finalImage);
  cv::imshow("draw_img", draw_img);
  cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
  cv::imshow("src", img);
  cv::setMouseCallback("src", mouse_callback, reinterpret_cast<void*>(&img));
  cv::waitKey(0);
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

void SiftFeature() {
  // 读入图像
  std::string path = "/home/crrcdt123/datasets2/二门防夹数据/车间/cam357-Start-20230110-141705915.jpg";
  cv::Mat img = cv::imread(path);

  // 转换为灰度图像
  cv::Mat grayImg;
  cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

  // 创建 SIFT 特征提取器
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

  // 检测关键点和计算描述符
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  sift->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

  // 在图像上绘制关键点
  cv::Mat image_with_keypoints;
  cv::drawKeypoints(img, keypoints, image_with_keypoints);

  // 显示图像和关键点
  cv::imshow("Image with Keypoints", image_with_keypoints);
  cv::waitKey(0);
  cv::destroyAllWindows();

}
void overlay() {
  std::vector<cv::String> image_paths;
  std::string root = "/home/crrcdt123/pytorch_test/pytorch-deeplab-xception/traindata/train_voc/";
  cv::glob("/home/crrcdt123/pytorch_test/pytorch-deeplab-xception/traindata/train_voc/JPEGImages/*.jpg", image_paths);
  for (size_t idx = 0; idx < image_paths.size(); idx+=10) {
    // std::string image_path = "/home/crrcdt123/pytorch_test/pytorch-deeplab-xception/traindata/train_voc/JPEGImages/frame_002662++frame_002673.jpg";
    std::string image_path = image_paths[idx];
    int pos = image_path.find_last_of('/');
    printf("total size: %d, sub size: %d\n", image_path.size(), pos);
    std::string name = image_path.substr(pos, image_path.size() - pos - 4) ;
    std::string mask_path = "/home/crrcdt123/pytorch_test/pytorch-deeplab-xception/traindata/train_voc/SegmentationClass"
     + name + ".png";

    cv::Mat image = cv::imread(image_path);
    cv::Mat mask = cv::imread(mask_path);

    if (mask.cols != image.cols || mask.rows != image.rows) {
      std::cout << "图像和掩码的尺寸不匹配:" << image.size() << "," << mask.size() << std::endl;
      return;
    }
    std::vector<cv::Point3i> pixes;
    for (size_t row = 0; row < mask.rows; row++) {
      for (size_t col = 0; col < mask.cols; col++) {
        cv::Point3i pix;
        pix.x = int(mask.at<cv::Vec3b>(row, col)[0]);
        pix.y = int(mask.at<cv::Vec3b>(row, col)[1]);
        pix.z = int(mask.at<cv::Vec3b>(row, col)[2]);
        if (pix.x == 1) {
          mask.at<cv::Vec3b>(row, col)[0] = 0;
          mask.at<cv::Vec3b>(row, col)[1] = 255;
          mask.at<cv::Vec3b>(row, col)[2] = 0;
        }
      }
    }

    cv::Mat overlay_image;
    cv::addWeighted(image, 0.7, mask, 0.3, 0.0, overlay_image);
    cv::imshow("image", overlay_image);
    cv::waitKey(100);
    cv::imwrite(root + "/overlay/" + name + ".jpg", overlay_image);
  }
}

