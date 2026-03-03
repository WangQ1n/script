#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderLargeImage.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <pcl/filters/passthrough.h>    
#include <pcl/common/transforms.h>

std::vector<cv::Point2f> projectPoints(const std::vector<cv::Point3f>& points, const cv::Mat& K, const cv::Mat& R, const cv::Mat& t);


int main(int argc, char** argv)
{
    // std::cout << cv::getBuildInformation() << std::endl;
    // std::vector<int> fourccs = {
    //     cv::VideoWriter::fourcc('X', '2', '6', '4'), // H.264
    //     cv::VideoWriter::fourcc('H', 'E', 'V', 'C'), // H.265 / HEVC
    //     cv::VideoWriter::fourcc('H', '2', '6', '5')  // Another H.265 option
    // };
    // std::string outputVideoFile1 = "output_video_h265.mp4";
    // std::string pipeline = "appsrc ! videoconvert ! x265enc ! matroskamux ! filesink location=" + outputVideoFile1;
    // for (int fourcc : fourccs) {
    //     cv::VideoWriter writer;
    //     bool success = writer.open(outputVideoFile1, fourcc, 25, cv::Size(640, 480));
    //     if (success) {
    //         std::cout << "Supported codec: " << cv::format("%c%c%c%c", fourcc & 0xFF, (fourcc >> 8) & 0xFF, (fourcc >> 16) & 0xFF, (fourcc >> 24) & 0xFF) << std::endl;
    //         writer.release();
    //     } else {
    //         std::cout << "Unsupported codec: " << cv::format("%c%c%c%c", fourcc & 0xFF, (fourcc >> 8) & 0xFF, (fourcc >> 16) & 0xFF, (fourcc >> 24) & 0xFF) << std::endl;
    //     }
    // }
    // return 1;

    std::string input_pcd_file = "/home/crrcdt123/1709783726.073115974.pcd";
    std::string output_image_file = "/home/crrcdt123/1709783726.073115974.png";;
    cv::Mat img = cv::imread(output_image_file);
    // 定义视频文件名、编码格式、帧率和视频尺寸
    std::string outputVideoFile = "/home/crrcdt123/output_video.avi";
    int codec = cv::VideoWriter::fourcc('X', 'V', 'I', 'D'); // MPEG-4 编码
    double fps = 10.0; // 帧率
    int width = 640;
    int height = 480;
    cv::Size frameSize(width, height);

    // 创建 VideoWriter 对象
    cv::VideoWriter videoWriter;
    videoWriter.open(outputVideoFile, cv::CAP_FFMPEG, codec, fps, frameSize, true);
    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not open the video file for writing." << std::endl;
        return -1;
    }

    // 加载点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(input_pcd_file, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file %s \n", input_pcd_file.c_str());
        return -1;
    }
    Eigen::Affine3f transformMatrix = pcl::getTransformation(3.76, -0.60, 0.0, 1.0 * M_PI / 180., 0.5 * M_PI / 180., 0);
    pcl::transformPointCloud(*cloud, *cloud, transformMatrix);
    pcl::PointCloud<pcl::PointXYZI>::Ptr roi_clouds(new pcl::PointCloud<pcl::PointXYZI>);
    // 创建滤波器对象
    pcl::PassThrough<pcl::PointXYZI> pass;      // 创建滤波器对象
    pass.setInputCloud(cloud);                 // 设置待滤波的点云
    pass.setFilterFieldName("z");               // 设置在Z轴方向上进行滤波
    pass.setFilterLimits(20, 30); // 设置滤波范围为0~1,在范围之外的点会被剪除
    // pass.setFilterLimitsNegative (true);//是否反向过滤，默认为false
    pass.filter(*roi_clouds); // 开始过滤

    // 创建 PCL 可视化器（无需显示窗口）
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer", true));
    // 获取 VTK 渲染窗口
    vtkSmartPointer<vtkRenderWindow> renderWindow = viewer->getRenderWindow();
    renderWindow->OffScreenRenderingOn(); // 启用离屏渲染
    
    viewer->setBackgroundColor(0.01, 0.01, 0.01);
    viewer->initCameraParameters();

    // 设置相机位置和方向
    pcl::visualization::Camera camera;
    camera.pos[0] = 4.52881;
    camera.pos[1] = -0.150151;
    camera.pos[2] = -4.84925;
    camera.view[0] = 0.999912;
    camera.view[1] = 0.00951673;
    camera.view[2] = 0.00929289;
    camera.focal[0] = 4.44688;
    camera.focal[1] = -0.187977;
    camera.focal[2] = 4.00493;
    camera.clip[0] = 0.652612;
    camera.clip[1] = 652.612;
    camera.fovy = 49.1311 / 180. * M_PI;
    camera.window_size[0] = width;
    camera.window_size[1] = height;
    viewer->setCameraParameters(camera);
    int len = 10000;
        auto start = std::chrono::steady_clock::now();
        // 清除以前的点云
        viewer->removeAllPointClouds();
        viewer->addPointCloud<pcl::PointXYZI>(cloud, "ori cloud");
        viewer->addPointCloud<pcl::PointXYZI>(roi_clouds, "roi cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "ori cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "roi cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "roi cloud");
        // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
        // viewer->spin();
        // return 1;

        auto vtk_start = std::chrono::steady_clock::now();
        // 渲染大图像（在内存中渲染，而不是在窗口中显示）
        vtkSmartPointer<vtkRenderLargeImage> renderLargeImage = vtkSmartPointer<vtkRenderLargeImage>::New();
        renderLargeImage->SetInput(viewer->getRendererCollection()->GetFirstRenderer());
        renderLargeImage->SetMagnification(1); // 设置放大倍率以提高图像分辨率

        try {
            renderLargeImage->Update();
        } catch (const std::exception& e) {
            std::cerr << "Error during rendering: " << e.what() << std::endl;
            return 1;
        }
        auto end = std::chrono::steady_clock::now();
        // 将渲染的图像保存为 PNG 文件
        // vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
        // writer->SetFileName(output_image_file.c_str());
        // writer->SetInputConnection(renderLargeImage->GetOutputPort());
        // writer->Write();
        vtkSmartPointer<vtkImageData> imageData = renderLargeImage->GetOutput();
        int* dims = imageData->GetDimensions();
        cv::Mat frame(dims[1], dims[0], CV_8UC3, imageData->GetScalarPointer());
        cv::flip(frame, frame, 0);
        cv::imshow("img", frame);
        cv::waitKey(1);
        // 写入视频文件
    while (len--)
    {   
        videoWriter.write(frame);
        // printf("cost time:%f, %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(vtk_start - start).count() / 1000.,
        // std::chrono::duration_cast<std::chrono::microseconds>(end - vtk_start).count() / 1000.);
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    videoWriter.release();
    return 0;
}


// 定义投影函数
std::vector<cv::Point2f> projectPoints(const std::vector<cv::Point3f>& points, const cv::Mat& K, const cv::Mat& R, const cv::Mat& t) {
    // 将3D点转换为齐次坐标
    cv::Mat pointsMat(points.size(), 3, CV_32F);
    for (size_t i = 0; i < points.size(); ++i) {
        pointsMat.at<float>(i, 0) = points[i].x;
        pointsMat.at<float>(i, 1) = points[i].y;
        pointsMat.at<float>(i, 2) = points[i].z;
    }
    
    // 投影矩阵
    cv::Mat P = K * (R * cv::Mat::eye(3, 3, CV_32F) + t);
    
    // 投影3D点
    cv::Mat projectedPointsMat;
    cv::projectPoints(pointsMat, R, t, K, cv::Mat(), projectedPointsMat);
    
    // 转换为2D点
    std::vector<cv::Point2f> projectedPoints;
    for (int i = 0; i < projectedPointsMat.rows; ++i) {
        projectedPoints.emplace_back(projectedPointsMat.at<float>(i, 0), projectedPointsMat.at<float>(i, 1));
    }
    
    return projectedPoints;
}