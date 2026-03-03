#include <opencv2/opencv.hpp>

int flaw_detect();
int clahe();

int main() {
  clahe();
  return 0;
  cv::Mat img = cv::Mat::zeros(cv::Size(1920, 1080), CV_8UC3);
  cv::Rect rect(224, 1030, 1200, 50);
  cv::Mat img2;
  img2 = img(rect);
  cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
  cv::imshow("kkk", img);
  cv::waitKey(0);
  // 创建两个矩形
  cv::Rect rect1(10, 10, 50, 50);  // (x, y, width, height)
  cv::Rect rect2(70, 30, 40, 50);

  // 计算交集区域
  cv::Rect intersection = rect1 & rect2;

  // 输出结果
  std::cout << "Intersection Area: " << intersection.area() << std::endl;
  std::cout << "Intersection Rect: " << intersection << std::endl;

  ////////////////////////////////////////////////////
  std::vector<cv::Point> contour;
  contour.emplace_back(cv::Point(1200, 1079));
  contour.emplace_back(cv::Point(500, 600));
  contour.emplace_back(cv::Point(1021, 930));
  // contour.emplace_back(cv::Point(1200, 1079));
  cv::Rect bbox = cv::boundingRect(contour);
  cv::Mat mask = cv::Mat::zeros(bbox.height, bbox.width, CV_8UC1);
  cv::fillPoly(mask, contour, 255, 8, 0, cv::Point(-bbox.x - 10, -bbox.y));
  return 0;
}

int clahe() {
   // 读取图像
    cv::Mat image = cv::imread("/home/crrcdt123/二门数据/1012_delay_1.5/cam113-End-20241009-062646437.jpg", cv::IMREAD_GRAYSCALE);  // 读取为灰度图像
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // 创建 CLAHE 对象
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);  // 对比度限制
    clahe->setTilesGridSize(cv::Size(8, 8));  // 局部区域大小

    // 应用 CLAHE
    cv::Mat clahe_image;
    clahe->apply(image, clahe_image);

    // 显示结果
    cv::imshow("Original Image", image);
    cv::imshow("CLAHE Image", clahe_image);

    cv::waitKey(0);
    return 0;
}
int flaw_detect() {
// Read the image
    cv::Mat img = cv::imread("/home/crrcdt123/Downloads/1.bmp");
    
    if (img.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    // Display the original image
    cv::imshow("src", img);

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Apply median blur with kernel size 201
    cv::Mat mean;
    cv::medianBlur(gray, mean, 201);

    // Display the mean image
    cv::imshow("mean", mean);

    // Compute the difference between the grayscale and mean images
    cv::Mat diff = gray - mean;

    // Display the difference image
    cv::imshow("diff", diff);

    // Save the difference image
    // cv::imwrite("diff.jpg", diff);

    // Wait for a key press and close the windows
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}