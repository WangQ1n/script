#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <opencv4/opencv2/opencv.hpp>
#include <filesystem> // C++17 中的标准库
void ViewerClear();
void ShowClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);
Eigen::Isometry3f GetT(std::vector<double> rpy, std::vector<double> t);
void filterPointCloudVoxel(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut);
void birdeyeview(pcl::PointCloud<pcl::PointXYZ>::Ptr clouds,
                 cv::Mat& bev_image);
std::string ExtractFileName(const std::string& path);
pcl::visualization::PCLVisualizer::Ptr visualizer_(
  new pcl::visualization::PCLVisualizer("PointCloud Visualizer"));
int cloud_range_ = 100;
int image_range_ = 1000;
int main() {
  visualizer_->setCameraPosition(3.67156, -0.484099, -13.336, 0.993612,
                                 0.00901104, 0.112489);
  visualizer_->setBackgroundColor(0, 0, 0);
  visualizer_->getRenderWindow()->GlobalWarningDisplayOff();  // Add This Line
  std::vector<cv::String> data_dirs = {
      "/workspace/rail_lidar_perception/data/*.pcd",
      "/home/crrcdt123/轨道点云数据/弯道点云/*.pcd",
      "/home/crrcdt123/轨道点云数据/弯道直线/*.pcd",
      "/media/crrcdt123/glam/crrc/datasets/点云数据/轨道数据/INNO-PCD/FXDD-FDCSGC/172.168.1.10/"
      "pcd/*.pcd",
      "/media/crrcdt123/glam/crrc/datasets/点云数据/轨道数据/INNO-PCD/QCY-XXDS/172.168.1.10/"
      "pcd/*.pcd",
      "/media/crrcdt123/glam/crrc/datasets/点云数据/轨道数据/INNO-PCD/KongPao-8s0/172.168.1.10/pcd/*.pcd",
      "/media/crrcdt123/glam/crrc/datasets/s8/lidar/pcd_train/*.pcd",
      "/media/crrcdt123/glam/crrc/datasets/s8/lidar/pcd_test/*.pcd",
      "/media/crrcdt123/glam/crrc/datasets/s8/lidar/pcd_obstacle/*.pcd"};
  std::vector<cv::String> data_paths;
  cv::glob(data_dirs[8], data_paths);
  std::cout << "总计点云帧数：" << data_paths.size() << std::endl;
  for (size_t idx = 0; idx < data_paths.size(); idx += 1) {
    // data_paths[idx] = "/home/crrcdt123/INNO_IDX_93225_TIME_1685989458.469675.pcd";
    std::string name = "0000" + std::to_string(idx);
    name = name.substr(name.length() - 4);
    pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);  // 创建点云指针。
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(data_paths[idx], *src_cloud) == -1) {
      std::cout << "Couldn't read file test_pcd_file.pcd" << std::endl;
      continue;
    } else {
      std::cout << "data path:" << idx << ", " << data_paths[idx] << std::endl;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr clouds(
        new pcl::PointCloud<pcl::PointXYZ>);
    // std::vector<double> rpy = {2.5, 4.3, -0.013}; //FXXD
    // std::vector<double> rpy = {1.5001052, -2.6999984, 0.0000044}; // kongpao
    // std::vector<double> t = {2.4, -0.8, 0.0};
    std::vector<double> rpy = {1.0, 0.5, 0}; // s8
    std::vector<double> t = {3.76, -0.60, 0.0}; //s8
    Eigen::Isometry3f lidar_T = GetT(rpy, t);
    std::cout << lidar_T.matrix() << std::endl;
    Eigen::Affine3f transformMatrix = pcl::getTransformation(3.76, -0.60, 0.0, 1.0 * M_PI / 180., 0.5 * M_PI / 180., 0);
    std::cout << transformMatrix.matrix() << std::endl;
    pcl::transformPointCloud(*src_cloud, *src_cloud, transformMatrix);
    for (auto point : src_cloud->points) {
      if (point.x == 0 && point.y == 0 && point.z == 0) continue;
      if (point.x < -2.5 || point.x > 2.5) continue;
      clouds->points.emplace_back(point);
    }
    filterPointCloudVoxel(clouds, clouds);
    cv::Mat bev_image, processed_img;
    birdeyeview(clouds, bev_image);
    cv::imshow("bev", bev_image);
    cv::waitKey(10);
    ViewerClear();
    ShowClouds(clouds);
    std::string fileName = ExtractFileName(data_paths[idx]);
    // filePath.
    // cv::imwrite("/media/crrcdt123/glam/crrc/datasets/点云数据/轨道数据/INNO-PCD/FXDD-FDCSGC/172.168.1.10/images0.2/" + fileName + ".jpg", bev_image);
    // cv::imwrite("/media/crrcdt123/glam/crrc/datasets/点云数据/轨道数据/INNO-PCD/KongPao-8s0/172.168.1.10/images/" + fileName + ".jpg", bev_image);
    cv::imwrite("/media/crrcdt123/glam/crrc/datasets/s8/lidar/pcd_obstacle/images/" + fileName + ".jpg", bev_image);
    // pcl::io::savePCDFileASCII("/home/crrcdt123/output_cloud.pcd", *src_cloud);
  }
}

void birdeyeview(pcl::PointCloud<pcl::PointXYZ>::Ptr clouds,
                 cv::Mat& bev_image) {
  // 点云范围y[-2, 2], z[0, 200] -> 图像范围y[0, 200], x[1080, 0]
  cv::Size image_size(image_range_, image_range_);
  cv::Size cloud_size(cloud_range_, cloud_range_);
  cv::Mat bev_img(image_size, CV_8UC1, cv::Scalar(0));
  float scale_ration = image_size.height / float(cloud_size.height);
  std::vector<cv::Point> points;
  // cv::cvtColor(bev_image, bev_image, cv::COLOR_GRAY2BGR);
  for (auto point : clouds->points) {
    if (point.z > cloud_range_) {
      continue;
    }
    // 1, 
    float z_ration = (cloud_range_ - point.z) / cloud_range_;
    int x = point.y * scale_ration * 5;
    int y = point.z * scale_ration * 1;
    if (x > image_range_ || y > image_range_) {
      continue;
    }
    x = x + image_size.width / 2;
    y = image_size.height - y;
    float plane_dist = point.x + 2.5;
    // bev_img.at<uchar>(y, x) =
    //     std::min(int(std::max(double(plane_dist)*70, 0.)), 255);
    cv::circle(bev_img, cv::Point(x,y), 2, std::min(int(std::max(double(plane_dist)*70, 0.)), 255), -1);
  }
  cv::applyColorMap(bev_img, bev_image, cv::COLORMAP_JET); // 使用 JET 色彩映射
  // bev_image = bev_img.clone();
}

void filterPointCloudVoxel(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut) {
  pcl::VoxelGrid<pcl::PointXYZ> voxelFilter;
  voxelFilter.setLeafSize(6, 0.05, 0.05);
  voxelFilter.setInputCloud(cloudIn);
  voxelFilter.filter(*cloudOut);
}

Eigen::Isometry3f GetT(std::vector<double> rpy, std::vector<double> t) {
  Eigen::Vector3d rpy_raw =
      Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
          rpy.data(), 3, 1);
  Eigen::Vector3d t_raw =
      Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(t.data(),
                                                                       3, 1);
  rpy_raw = rpy_raw * M_PI / 180;
  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(rpy_raw[2], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(rpy_raw[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(rpy_raw[0], Eigen::Vector3d::UnitX());
  Eigen::Isometry3f T = Eigen::Isometry3f::Identity();  // 变换矩阵
  T.rotate(R.cast<float>());
  T.pretranslate(t_raw.cast<float>());
  return T;
}

void ViewerClear() {
  visualizer_->removeText3D();
  visualizer_->removeAllPointClouds();
  visualizer_->removeAllShapes();
}

void ShowClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
  visualizer_->addPointCloud<pcl::PointXYZ>(point_cloud, "PointCloud");
  visualizer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud");
  visualizer_->addCoordinateSystem(5.0);

  visualizer_->spinOnce(10);
  // std::this_thread::sleep_for(std::chrono::milliseconds(100));
  // visualizer_->removeAllPointClouds();
  // visualizer_->removeAllShapes();
}

std::string ExtractFileName(const std::string& path) {
    // 使用 '/' 或 '\\' 分割路径
    std::istringstream iss(path);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(iss, token, '/') || std::getline(iss, token, '\\')) {
        tokens.push_back(token);
    }

    // 提取最后一个字符串作为文件名
    std::string fileName = tokens.back();

    // 移除文件名中的后缀
    size_t dotIndex = fileName.find_last_of('.');
    if (dotIndex != std::string::npos) {
        fileName = fileName.substr(0, dotIndex);
    }

    return fileName;
}
