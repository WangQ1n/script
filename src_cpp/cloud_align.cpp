// 测试不同配准方法效果
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/visualization/pcl_visualizer.h>
using PointType = pcl::PointXYZ;
using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = PointCloudType::Ptr;
inline CloudPtr VoxelCloud(CloudPtr cloud, float voxel_size = 0.1) {
  CloudPtr output(new PointCloudType);
  pcl::VoxelGrid<PointType> voxel;                        // 体素滤波器
  voxel.setLeafSize(voxel_size, voxel_size, voxel_size);  // 设置分辨率
  voxel.setInputCloud(cloud);                             // 设置待滤波的输入点云
  voxel.filter(*output);  // 执行滤波，将滤波结果存储到output中
  return output;
}
const std::vector<std::vector<double>> color{
    {1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}, {0.5, 0.5, 0.}, {0., 0.5, 0.5}};

void computeGuessPose(pcl::PointCloud<pcl::PointXYZ>::Ptr map, pcl::PointCloud<pcl::PointXYZ>::Ptr scan,
                      Eigen::Matrix4f& T) {
  // 计算法线
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(scan);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  ne.setSearchMethod(tree);
  pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud<pcl::Normal>);
  ne.setRadiusSearch(0.1);
  ne.compute(*source_normals);

  ne.setInputCloud(map);
  pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud<pcl::Normal>);
  ne.compute(*target_normals);

  // 计算FPFH特征
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_fpfh(new pcl::PointCloud<pcl::FPFHSignature33>());
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_fpfh(new pcl::PointCloud<pcl::FPFHSignature33>());

  fpfh.setInputCloud(scan);
  fpfh.setInputNormals(source_normals);
  fpfh.setSearchMethod(tree);
  fpfh.setRadiusSearch(0.1);
  fpfh.compute(*source_fpfh);

  fpfh.setInputCloud(map);
  fpfh.setInputNormals(target_normals);
  fpfh.compute(*target_fpfh);

  // 使用SAC-IA进行粗配准
  pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_prerejective;
  sac_prerejective.setInputSource(scan);
  sac_prerejective.setSourceFeatures(source_fpfh);
  sac_prerejective.setInputTarget(map);
  sac_prerejective.setTargetFeatures(target_fpfh);
  sac_prerejective.setMaximumIterations(50000);         // 最大迭代次数
  sac_prerejective.setNumberOfSamples(3);               // 样本数
  sac_prerejective.setCorrespondenceRandomness(5);      // 随机选取对应点
  sac_prerejective.setSimilarityThreshold(0.9f);        // 相似性阈值
  sac_prerejective.setMaxCorrespondenceDistance(0.2f);  // 对应点最大距离
  sac_prerejective.setInlierFraction(0.25f);            // 内点比例

  pcl::PointCloud<pcl::PointXYZ>::Ptr sac_ia_output(new pcl::PointCloud<pcl::PointXYZ>());
  sac_prerejective.align(*sac_ia_output);

  Eigen::Matrix4f sac_transformation = sac_prerejective.getFinalTransformation();
  std::cout << "SAC-IA Transformation: \n" << sac_transformation << std::endl;
  T = sac_transformation;
}
void NdtAlign(pcl::PointCloud<pcl::PointXYZ>::Ptr map, pcl::PointCloud<pcl::PointXYZ>::Ptr scan,
              Eigen::Matrix4f T) {
  pcl::visualization::PCLVisualizer viewer("NDT Registration");
  // 创建 NDT 对象
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  ndt.setTransformationEpsilon(0.000005);
  ndt.setMaximumIterations(35);  // 设置最大迭代次数
  ndt.setStepSize(1.5);          // 设置步长

  std::cout << "start align" << std::endl;
  pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
  // ndt.setResolution(1.0);  // 分辨率
  // ndt.setInputSource(scan);
  // ndt.setInputTarget(map);
  //   ndt.align(aligned_cloud, T);  // 对齐
  std::vector<double> res{2.0, 1.0};  // 四种体素分辨率 分辨率超过2就不行了，怀疑室内场景狭小不适合
  int idx = 0;
  for (auto& r : res) {
    auto rough_map = VoxelCloud(map, r * 0.1);
    auto rough_scan = VoxelCloud(scan, r * 0.1);  // 滤波后的子地图2
    ndt.setInputTarget(map);
    ndt.setInputSource(scan);
    ndt.setResolution(r);
    ndt.align(aligned_cloud, T);
    // 将上一个粗配准结果代入下一次配准中
    // T = ndt.getFinalTransformation();
    // 输出配准结果
    std::cout << "NDT converged: " << ndt.hasConverged() << ", score: " << ndt.getFitnessScore()
              << ", score2:" << ndt.getTransformationProbability() << std::endl;
    std::cout << "Transformation matrix:\n" << ndt.getFinalTransformation() << std::endl;
    viewer.addPointCloud(aligned_cloud.makeShared(), "aligned cloud " + std::to_string(r));
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[idx][0],
                                            color[idx][1], color[idx][2],
                                            "aligned cloud " + std::to_string(r));  // 红色
    idx++;
  }

  // 可视化结果（可选）

  viewer.setBackgroundColor(0, 0, 0);
  viewer.addPointCloud(map, "target cloud");
  viewer.addPointCloud(scan, "source cloud");
  // std::cout << "NDT converged: " << ndt.hasConverged() << ", score: " << ndt.getFitnessScore() <<
  // ", score2:" << ndt.getTransformationProbability() << std::endl; std::cout << "Transformation matrix:\n"
  // << ndt.getFinalTransformation() << std::endl;
  // viewer.addPointCloud(aligned_cloud.makeShared(), "aligned cloud"); 
  // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
  // color[idx][0], color[idx][1], color[idx][2],
  //                                         "aligned cloud");  // 红色
  viewer.spin();
}

void ICPAlign(pcl::PointCloud<pcl::PointXYZ>::Ptr map, pcl::PointCloud<pcl::PointXYZ>::Ptr scan,
              Eigen::Matrix4f T) {
  // 创建 ICP 对象
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setMaximumIterations(35);        // 设置最大迭代次数
  icp.setTransformationEpsilon(1e-4);  // 设置收敛条件
  //   icp.setEuclideanFitnessEpsilon(1);   // 设置拟合误差容忍度

  // 设置目标和源点云
  icp.setInputSource(scan);
  icp.setInputTarget(map);

  // 执行配准
  pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
  icp.align(aligned_cloud, T);  // 对齐

  // 输出配准结果
  std::cout << "ICP converged: " << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
  std::cout << "Final transformation:\n" << icp.getFinalTransformation() << std::endl;

  // 可视化结果（可选）
  pcl::visualization::PCLVisualizer viewer("ICP Registration");
  viewer.setBackgroundColor(0, 0, 0);
  viewer.addPointCloud(map, "target cloud");
  viewer.addPointCloud(scan, "source cloud");
  viewer.addPointCloud(aligned_cloud.makeShared(), "aligned cloud");
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0][0],
                                          color[0][1], color[0][2],
                                          "aligned cloud");  // 红色
  viewer.spin();
}
void testCostTime(pcl::PointCloud<pcl::PointXYZ>::Ptr map) {
  pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(map);
}

int main(int argc, char** argv) {
  // 创建点云指针
  pcl::PointCloud<pcl::PointXYZ>::Ptr map(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr scan(new pcl::PointCloud<pcl::PointXYZ>());

  // 加载目标点云（局部地图）
  if (pcl::io::loadPCDFile("/home/crrcdt123/git/slam_obstacle_detector/map/20250313-231437/00187.pcd", *map) == -1) {
    PCL_ERROR("Couldn't read target.pcd \n");
    return -1;
  }

  // 加载源点云（当前帧）
  if (pcl::io::loadPCDFile("/home/crrcdt123/git/slam_obstacle_detector/map/20250313-231437/00188.pcd", *scan) == -1) {
    PCL_ERROR("Couldn't read source.pcd \n");
    return -1;
  }
  map = VoxelCloud(map, 0.1);
  scan = VoxelCloud(scan, 0.1);          // 滤波后的子地图2
  // pcl::PassThrough<pcl::PointXYZ> pass;  // 创建滤波器对象
  // pass.setFilterFieldName("z");          // 设置在Z轴方向上进行滤波
  // pass.setFilterLimits(-1, 7);           // 设置滤波范围为0~1,在范围之外的点会被剪除
  // pass.setInputCloud(scan);              // 设置待滤波的点云
  // pass.filter(*scan);                    // 开始过滤
  // pass.setInputCloud(map);               // 设置待滤波的点云
  // pass.filter(*map);
  //   pass.setFilterFieldName("x");          // 设置在Z轴方向上进行滤波
  //   pass.setFilterLimits(-20, 5);           // 设置滤波范围为0~1,在范围之外的点会被剪除
  //   pass.setInputCloud(map);               // 设置待滤波的点云
  //   pass.filter(*map);                     // 开始过滤
  //     pass.setFilterFieldName("y");          // 设置在Z轴方向上进行滤波
  //   pass.setFilterLimits(-40, -10);           // 设置滤波范围为0~1,在范围之外的点会被剪除
  //   pass.setInputCloud(map);               // 设置待滤波的点云
  //   pass.filter(*map);                     // 开始过滤
  Eigen::AngleAxisd yaw_angle = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd roll_angle = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitch_angle = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY());
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();  // 初始化变换矩阵
  T.block<3, 1>(0, 3) = Eigen::Vector3f(0, 0, 0);  // 2米以内可以配准
  T.block<3, 3>(0, 0) = (yaw_angle * roll_angle * pitch_angle).toRotationMatrix().cast<float>();
  //   computeGuessPose(map, scan, T);
  ICPAlign(map, scan, T);
  // ndt 只能测试2米左右，90度以内配准OK，和icp差不多，无法测试出远距离.远了陷入局部最优解
  // NdtAlign(map, scan, T);

  return 0;
}
