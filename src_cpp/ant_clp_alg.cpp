#include "ant_clp_alg.h"

#include <chrono>
#include <iostream>
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
  //
  cv::cvtColor(prevSrcImg_, prevHsvImg_, cv::COLOR_BGR2HSV);
  // pre
  pre_process(prevSrcImg_, prevPreProImg_);
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
  //
  cv::cvtColor(currSrcImg_, currHsvImg_, cv::COLOR_BGR2HSV);
  // pre
  pre_process(currSrcImg_, currPreProImg_);

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

  // gray
  cv::cvtColor(inImg, outImg, cv::COLOR_BGR2GRAY);
  // equ
  // cv::equalizeHist
  // fileter gaussian noise
  // cv::GaussianBlur(outImg, outImg, cv::Size(7, 7), (0, 0));
  cv::blur(outImg, outImg, cv::Size(7, 7));
  // DEBUG_SHOW("outImg", outImg);
}

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
  image_segmentation(prevPreProImg_, currPreProImg_, maskImg_);
  DEBUG_SHOW("segmentation_mask", maskImg_);
  // 二值前景过滤  形态学处理
  morphphological_process(maskImg_);
  DEBUG_SHOW("maskImg0", maskImg_);

  // SIFT
  //  cv::Mat mask = maskImg_.clone();
  //  cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
  //  cv::erode(maskImg_, maskImg_, element, cv::Point(-1, -1));
  //  DEBUG_SHOW("erode maskImg_", maskImg_);
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
  advance_process(prevHsvImg_, currHsvImg_, maskImg_);
  DEBUG_SHOW("maskImg1", maskImg_);
  // 二值前景过滤  形态学处理
  //  morphphological_process(maskImg_);
  //  DEBUG_SHOW("maskImg2", maskImg_);
  // 标记
  std::vector<cv::Rect> vBoundRect;
  find_objdect(maskImg_, vBoundRect);

  std::vector<int> picked;
  box_filter(vBoundRect, picked, NMS_THRESH);

  // 画矩形框
  currSrcImg_.copyTo(resultImg);
  cv::cvtColor(maskImg_, maskImg_, cv::COLOR_GRAY2BGR);
  if (!vBoundRect.empty()) {
    for (auto it = picked.begin(); it < picked.end(); it++) {
      cv::rectangle(resultImg, vBoundRect[*it], cv::Scalar(0, 0, 255), 2);
    }
  }
  // DEBUG_SHOW("nms_object", maskImg_);
  for (auto it = vBoundRect.begin(); it < vBoundRect.end(); it++) {
    cv::rectangle(maskImg_, *it, cv::Scalar(0, 0, 255), 2);
  }
  // DEBUG_SHOW("object", maskImg_);
  // 返回检测结果
  if (result) {
    *result = picked.size();
  }

  // 释放图像资源 TODO
  if (IsShow) {
    cv::waitKey(0);
  }
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
void AntClpAlg::image_segmentation(const cv::Mat& img1, const cv::Mat& img2,
                                   cv::Mat& maskImg) {
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

        int dis = abs(grayImage2.at<uchar>(row, col) -
                      grayImage1.at<uchar>(neiborRow, neiborCol));
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
  }    // end for row
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
    std::cout << "Morphphological process: invalid input" << std::endl;
    return;
  } else {
    std::cout << "maskImg type:" << maskImg.type() << std::endl;
  }

  // 腐蚀与膨胀 TODO
  cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::erode(maskImg, maskImg, element, cv::Point(-1, -1));
  cv::dilate(maskImg, maskImg, element, cv::Point(-1, -1));

  std::vector<std::vector<cv::Point>> contours;  // 轮廓
  std::vector<cv::Vec4i> hierarchy;

  // cv::RETR_TREE检测所有轮廓
  // cv::CHAIN_APPROX_NONE 边界所有连续点
  cv::findContours(maskImg, contours, hierarchy, cv::RETR_TREE,
                   cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
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
        if (contArea <= 20) {
          cv::drawContours(maskImg, contours, contIndex, cv::Scalar(255));
        }
      }
    }

    else if (firstLevel == -1) {
      // 过滤面积过小的轮廓 TODO
      if (contArea <= 20 || perimeter <= 20) {
        cv::drawContours(maskImg, contours, contIndex, cv::Scalar(0));
      }  // end ContArea
    }    //
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
void AntClpAlg::find_objdect(const cv::Mat& inImg,
                             std::vector<cv::Rect>& vBoundRect) {
  if (inImg.empty()) {
    std::cout << "Find object invalid input" << std::endl;
    return;
  }

  std::vector<std::vector<cv::Point>> contours;  // 轮廓
  std::vector<cv::Vec4i> hierarchy;

  // cv::RETR_EXTERNALj仅仅检测外部轮廓
  // cv::CHAIN_APPROX_NONE 边界所有连续点
  cv::findContours(inImg, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
  if (!contours.size()) {
    return;
  }

  // sort 从大到小 标准库函数 自定义排序函数
  std::sort(contours.begin(), contours.end(), sortFunc);

  //
  for (int contIndex = 0; contIndex < contours.size(); contIndex++) {
    double contArea = cv::contourArea(contours[contIndex]);
    // TODO
    if (contArea > 20)  //
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
void AntClpAlg::box_filter(std::vector<cv::Rect>& vBoundRect,
                           std::vector<int>& picked, float nms_threshold) {
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
      if (b.x <= a.x && b.y <= a.y && b.width > a.width &&
          b.height > a.height) {
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
void AntClpAlg::advance_process(cv::Mat& img1, cv::Mat& img2,
                                cv::Mat& maskImg) {
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
        if (hmin <= DEFAULT_HUE_THVAL && smin <= DEFAULT_SATURATION_THVAL &&
            vmin <= DEFAULT_VALUE_THVAL) {
          // 色度 "红", "橙", "黄", "绿", "青", "蓝", "紫"
          hsvRegionGrowing(img2, img1, maskImg, col, row, currentColor, 3, 20,
                           0);
          maskImg.at<uchar>(row, col) = 0;
          std::cout << col << ", " << row << ", " << currentColor << std::endl;
          std::cout << std::endl;
          // DEBUG_SHOW("colorRegionGrowing", maskImg);
          // cv::waitKey(0);
        }
      } else {
        if ((prevS >= 43 && currS >= 43) && (currV >= 27 && prevV > 27) &&
            hmin <= DEFAULT_HUE_THVAL && smin <= DEFAULT_SATURATION_THVAL &&
            vmin <= DEFAULT_VALUE_THVAL) {
          // 色度 "红", "橙", "黄", "绿", "青", "蓝", "紫"
          hsvRegionGrowing(img2, img1, maskImg, col, row, currentColor, 5, 30,
                           0);
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
void AntClpAlg::advance_process(cv::Mat& img1, cv::Mat& img2,
                                cv::Mat& maskImg) {
  cv::Mat diff, hsvmask;
  cv::absdiff(img1, img2, diff);
  cv::blur(diff, diff, cv::Size(7, 7));
  inRange(diff,
          cv::Scalar(DEFAULT_HUE_THVAL, DEFAULT_SATURATION_THVAL,
                     DEFAULT_VALUE_THVAL),
          cv::Scalar(180, 255, 255), hsvmask);
  morphphological_process(hsvmask);
  cv::bitwise_and(hsvmask, maskImg, maskImg);
  /*
          cv::Mat hsvImg1[3], hsvImg2[3];
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
      std::cout << "at(" << x << "," << y << ")value is:"
                << static_cast<int>(im->at<uchar>(cv::Point(x, y)))
                << std::endl;
      break;
    case 2:  // 鼠标右键按下响应：输入坐标并返回该坐标的灰度
      std::cout << "input(x,y)" << std::endl;
      std::cout << "x =" << std::endl;
      std::cin >> x;
      std::cout << "y =" << std::endl;
      std::cin >> y;
      std::cout << "at(" << x << "," << y << ")value is:"
                << static_cast<int>(im->at<uchar>(cv::Point(x, y)))
                << std::endl;
      break;
  }
}
// int main(int argc, char **argv)
// {
// 		AntClpAlg alg;
//     unsigned char findObjs = 0;
//     cv::Mat img1, img2, resultImg;
// 		std::string start_path =
// "/home/crrcdt123/git/test_func/data/二门/cam224-Start-20230110-094326575.jpg";
// 		std::string end_path =
// "/home/crrcdt123/git/test_func/data/二门/cam224-End-20230110-094337621.jpg";
//     img1 = cv::imread(start_path);
//     img2 = cv::imread(end_path);
//     img2.copyTo(resultImg);
//     //cv::Rect rect(465, 734, 400, 43);
//     cv::Rect rect(1200, 240, 300, 70);
//     //cv::Rect rect(275, 565, 183, 140);
//     //cv::Rect rect(275, 168, 400, 400);

//     std::chrono::steady_clock::time_point st, et;
//     st = std::chrono::steady_clock::now();
//     alg.input_before(img1(rect));
//     et = std::chrono::steady_clock::now();
//     printf("run time : %ld\n",
//     std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count());

//     st = std::chrono::steady_clock::now();
//     alg.input_after(img2(rect));
//     et = std::chrono::steady_clock::now();
//     printf("run time : %ld\n",
//     std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count());

//     st = std::chrono::steady_clock::now();
//     cv::Mat objimg;
//     alg.process(objimg, &findObjs );
//     printf("find objs: %d\n", findObjs);
//     et = std::chrono::steady_clock::now();
//     printf("run time : %ld\n",
//     std::chrono::duration_cast<std::chrono::milliseconds>(et -st).count());
//     //
//     if(findObjs > 0)
//     {
//         objimg.copyTo(resultImg(rect));
//     }
// 		namedWindow("src", cv::WINDOW_NORMAL);
// 		namedWindow("obj", cv::WINDOW_NORMAL);
// 		namedWindow("detect_result", cv::WINDOW_NORMAL);
// 		// cv::setMouseCallback("obj", onMouse, reinterpret_cast<void*>
// (&resultImg));//注册鼠标操作(回调)函数 		cv::rectangle(img1,
// rect, cv::Scalar(0, 255, 0), 2);
//     cv::imshow("src", img1);
// 		cv::rectangle(resultImg, rect, cv::Scalar(0, 255, 0), 2);
//     cv::imshow("obj", objimg);
//     cv::imshow("detect_result", resultImg);
//     cv::waitKey(0);

//     cv::destroyAllWindows();

//     return 0;

// }