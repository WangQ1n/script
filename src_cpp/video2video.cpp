#include <iostream>
#include <opencv2/opencv.hpp>


void OpencvVideo(cv::VideoCapture& cap, std::string outputVideoFile, int frameWidth, int frameHeight, double fps);
void FFmpegPopen(cv::VideoCapture& cap, std::string outputVideoFile, int frameWidth, int frameHeight, double fps);

int main() {
  // 打开视频文件
  std::string inputVideoFile = "/home/crrcdt123/0022-20240307-094906.mp4";  // 输入视频文件路径
  cv::VideoCapture cap(inputVideoFile);
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open the input video file." << std::endl;
    return -1;
  }

  // 获取视频属性
  int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);

  // 定义输出视频文件
  std::string outputVideoFile = "/home/crrcdt123/output_hevc.mp4";  // 输出视频文件路径

  OpencvVideo(cap, outputVideoFile, frameWidth, frameHeight, fps);

//   FFmpegPopen(cap, outputVideoFile, frameWidth, frameHeight, fps);

  return 0;
}

void OpencvVideo(cv::VideoCapture& cap, std::string outputVideoFile, int frameWidth, int frameHeight, double fps) {
  int codec = cv::VideoWriter::fourcc('H', 'E', 'V', 'C');          // H.265 编码格式

  // 创建 VideoWriter 对象
  cv::VideoWriter videoWriter(outputVideoFile, codec, fps, cv::Size(frameWidth,
  frameHeight), true);
  // std::cout <<  videoWriter.get(cv::VIDEOWRITER_PROP_FRAMEBYTES) << std::endl;
  //cv::VideoWriterProperties()
  // std::cout << videoWriter.set(1, 0.8) << std::endl;
  if (!videoWriter.isOpened()) {
      std::cerr << "Error: Could not open the output video file for writing." << std::endl;
      return;
  }

  cv::Mat frame;
  int length = 1000;
  while (length--) {
      // 读取一帧
      cap >> frame;
      if (frame.empty()) {
          break;  // 视频结束
      }

      // 写入帧到输出视频文件
      videoWriter.write(frame);
  }

  std::cout << "Video file has been written successfully." << std::endl;
}

void FFmpegPopen(cv::VideoCapture& cap, std::string outputVideoFile, int frameWidth, int frameHeight, double fps) {
    // 定义 FFmpeg 命令
  std::string ffmpeg_command = "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt bgr24 -s " +
                               std::to_string(frameWidth) + "x" + std::to_string(frameHeight) + " -r " +
                               std::to_string(fps) + " -i - -c:v libx265 -preset fast " +
                               outputVideoFile;

  // 打开 FFmpeg 管道
  FILE* ffmpeg = popen(ffmpeg_command.c_str(), "w");
  if (!ffmpeg) {
    std::cerr << "Error: Could not open FFmpeg pipe." << std::endl;
    return;
  }

  cv::Mat frame;
  int length = 1000;
  while (length--) {
    // 读取一帧
    cap >> frame;
    if (frame.empty()) {
      break;  // 视频结束
    }

    // 写入帧到 FFmpeg 管道
    fwrite(frame.data, 1, frame.total() * frame.elemSize(), ffmpeg);
  }

  // 关闭 FFmpeg 管道
  pclose(ffmpeg);

  std::cout << "Video file has been written successfully." << std::endl;
}
