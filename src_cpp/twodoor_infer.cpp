// tensorrt
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
// cuda
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "gpu_speed_test.h"
// __global__ void preprocess_batch_kernel(const uint8_t* __restrict__ imgs0,  // [N, H, W, 3]
//                                         const uint8_t* __restrict__ imgs1,  // [N, H, W, 3]
//                                         int N, int H, int W,
//                                         float* output,  // [N, 6, H, W]
//                                         const float* __restrict__ mean, const float* __restrict__ std) {
//   int x = threadIdx.x + blockIdx.x * blockDim.x;  // width
//   int y = threadIdx.y + blockIdx.y * blockDim.y;  // height
//   int n = blockIdx.z;                             // batch

//   if (x >= W || y >= H || n >= N) return;

//   int hw = H * W;
//   int img_area = H * W * 3;

//   // offset for current image
//   const uint8_t* img0 = imgs0 + n * img_area;
//   const uint8_t* img1 = imgs1 + n * img_area;

//   int in_idx = (y * W + x) * 3;

//   // 读取 img0 像素（BGR → RGB）
//   uint8_t b0 = img0[in_idx + 0];
//   uint8_t g0 = img0[in_idx + 1];
//   uint8_t r0 = img0[in_idx + 2];

//   // 读取 img1 像素
//   uint8_t b1 = img1[in_idx + 0];
//   uint8_t g1 = img1[in_idx + 1];
//   uint8_t r1 = img1[in_idx + 2];

//   // Normalize + 写入 CHW（通道顺序：R,G,B）
//   int base_out = n * 6 * hw;
//   int out_offset = y * W + x;

//   output[base_out + 0 * hw + out_offset] = (r0 / 255.0f - mean[0]) / std[0];
//   output[base_out + 1 * hw + out_offset] = (g0 / 255.0f - mean[1]) / std[1];
//   output[base_out + 2 * hw + out_offset] = (b0 / 255.0f - mean[2]) / std[2];

//   output[base_out + 3 * hw + out_offset] = (r1 / 255.0f - mean[0]) / std[0];
//   output[base_out + 4 * hw + out_offset] = (g1 / 255.0f - mean[1]) / std[1];
//   output[base_out + 5 * hw + out_offset] = (b1 / 255.0f - mean[2]) / std[2];
// }

// void preprocess_batch(const std::vector<cv::Mat>& img0_list, const std::vector<cv::Mat>& img1_list, int
// N,
//                       int H, int W, float* device_output) {
//   size_t single_img_bytes = H * W * 3;
//   size_t total_img_bytes = N * single_img_bytes;

//   // 分配 GPU 输入缓存
//   uint8_t *d_img0, *d_img1;
//   cudaMalloc(&d_img0, total_img_bytes);
//   cudaMalloc(&d_img1, total_img_bytes);

//   // 拼接所有图像数据到 CPU 内存
//   std::vector<uint8_t> cpu_img0_buf(total_img_bytes);
//   std::vector<uint8_t> cpu_img1_buf(total_img_bytes);
//   for (int i = 0; i < N; ++i) {
//     std::memcpy(cpu_img0_buf.data() + i * single_img_bytes, img0_list[i].data, single_img_bytes);
//     std::memcpy(cpu_img1_buf.data() + i * single_img_bytes, img1_list[i].data, single_img_bytes);
//   }

//   cudaMemcpy(d_img0, cpu_img0_buf.data(), total_img_bytes, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_img1, cpu_img1_buf.data(), total_img_bytes, cudaMemcpyHostToDevice);

//   // Mean / Std
// float mean[3] = {123.675, 116.28, 103.53};
// float std[3] = {58.395, 57.12, 57.375};
//   float *d_mean, *d_std;
//   cudaMalloc(&d_mean, 3 * sizeof(float));
//   cudaMalloc(&d_std, 3 * sizeof(float));
//   cudaMemcpy(d_mean, h_mean, 3 * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_std, h_std, 3 * sizeof(float), cudaMemcpyHostToDevice);

//   // 启动 Kernel
//   dim3 block(16, 16);
//   dim3 grid((W + 15) / 16, (H + 15) / 16, N);
//   preprocess_batch_kernel<< <grid, block>> >(d_img0, d_img1, N, H, W, device_output, d_mean, d_std);
//   cudaDeviceSynchronize();

//   // 清理
//   cudaFree(d_img0);
//   cudaFree(d_img1);
//   cudaFree(d_mean);
//   cudaFree(d_std);
// }

std::vector<char> readFile(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  size_t size = file.tellg();
  file.seekg(0);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  return buffer;
}

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
  }
} gLogger;

bool buildEngineFromOnnx(const std::string& onnx_path, nvinfer1::ICudaEngine* engine,
                         int maxBatch = 4) {
  using namespace nvinfer1;

  auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
  auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
  auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(
      1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

  auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));

  if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kINFO))) {
    std::cerr << "Failed to parse ONNX file." << std::endl;
    return false;
  }

  config->setMaxWorkspaceSize(1 << 28);  // 256MB
  if (builder->platformHasFastFp16()) config->setFlag(BuilderFlag::kFP16);

  auto profile = builder->createOptimizationProfile();
  auto input_name = network->getInput(0)->getName();
  Dims dims = network->getInput(0)->getDimensions();  // e.g., [-1, 3, 224, 224]
  profile->setDimensions(input_name, OptProfileSelector::kMIN, Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
  profile->setDimensions(input_name, OptProfileSelector::kOPT,
                         Dims4{2, dims.d[1], dims.d[2], dims.d[3]});
  profile->setDimensions(input_name, OptProfileSelector::kMAX,
                         Dims4{maxBatch, dims.d[1], dims.d[2], dims.d[3]});
  config->addOptimizationProfile(profile);

  engine = builder->buildEngineWithConfig(*network, *config);
  if (!engine) return false;

  // Save engine to file
  nvinfer1::IHostMemory* serialized_engine = engine->serialize();
  if (serialized_engine == nullptr) {
    printf("Serialized engine failed!");
    return false;
  }
  std::string engine_file_path = "/home/crrcdt123/git/script/twodoor.engine";
  std::ofstream p(engine_file_path, std::ios::binary);
  if (!p) {
    return false;
  }
  p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
  return true;
}

void infer(std::unique_ptr<nvinfer1::IExecutionContext>& context, void** bindings) {
  using namespace nvinfer1;
  context->enqueueV2(bindings, 0, nullptr);
  cudaDeviceSynchronize();
}

// 读取 trt 文件为字节数组
std::vector<char> loadEngineFile(const std::string& engineFile) {
  std::ifstream file(engineFile, std::ios::binary);
  if (!file) throw std::runtime_error("Failed to open engine file.");

  file.seekg(0, std::ifstream::end);
  size_t size = file.tellg();
  file.seekg(0, std::ifstream::beg);

  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  return buffer;
}

// 加载引擎
nvinfer1::ICudaEngine* loadEngine(const std::string& engineFile) {
  std::ifstream file(engineFile, std::ios::binary);
  if (!file) throw std::runtime_error("Failed to open engine file.");

  file.seekg(0, std::ifstream::end);
  size_t size = file.tellg();
  file.seekg(0, std::ifstream::beg);

  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
  return runtime->deserializeCudaEngine(buffer.data(), buffer.size());
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

void Preprocess(std::vector<cv::Mat> images1, std::vector<cv::Mat> images2, float* output, cudaStream_t stream){
  float mean[3] = {123.675, 116.28, 103.53};
  float std[3] = {58.395, 57.12, 57.375};
  float *d_mean, *d_std;
  cudaMalloc(&d_mean, 3 * sizeof(float));
  cudaMalloc(&d_std, 3 * sizeof(float));
  cudaMemcpy(d_mean, mean, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_std, std, 3 * sizeof(float), cudaMemcpyHostToDevice);
  int last_byte = 0;
  for (size_t i = 0; i < images1.size(); i++) {
    int width = images1[i].cols;
    int height = images1[i].rows;
    int single_byte = height * width * 3 * sizeof(uint8_t);
    uint8_t *d_img1, *d_img2;
    cudaMalloc(&d_img1, width * height * 3 * sizeof(uint8_t));
    cudaMalloc(&d_img2, width * height * 3 * sizeof(uint8_t));
    cudaMemcpy(d_img1, images1[i].data, single_byte, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, images2[i].data, single_byte, cudaMemcpyHostToDevice);
    preprocess(d_img1, d_img2, output + last_byte, height, width, 224, 224, d_mean, d_std, stream);
    last_byte = 224 * 224 * 6;
    cudaStreamSynchronize(stream);
    cudaFree(d_img1);
    cudaFree(d_img2);
  }
  cudaFree(d_mean);
  cudaFree(d_std);
}

int main() {
  cv::Mat img11, img12, img21, img22;
  std::vector<cv::Mat> images1;  // 每张图为CV_8UC3
  std::vector<cv::Mat> images2;  // 每张图为CV_8UC3

  float* device_input;
  int batch_size = 2;
  int out_h = 224, out_w = 224;

  nvinfer1::ICudaEngine* engine = nullptr;
  // if (!buildEngineFromOnnx("/home/crrcdt123/git/Siamese-pytorch/model.onnx", engine)) {
  //   return -1;
  // }
  engine = loadEngine("/home/crrcdt123/git/script/twodoor.engine");
  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
  if (!context) {
    std::cerr << "Failed to create context." << std::endl;
    return -1;
  }
  int inputIndex = engine->getBindingIndex("input");  // 更换为你模型输入名
  int outputIndex = engine->getBindingIndex("output");
  cudaMalloc(&device_input, batch_size * 6 * out_h * out_w * sizeof(float));
  // 设置动态 shape
  context->setBindingDimensions(inputIndex, nvinfer1::Dims4{batch_size, 6, out_h, out_w});

  float* output_d;
  size_t outputSize = engine->getBindingDimensions(outputIndex).d[1] * sizeof(float);  // 示例：分类模型
  cudaMalloc(&output_d, batch_size * outputSize);
  void* bindings[2] = {device_input, output_d};

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  int cnt = 0;
  std::string root = "/home/crrcdt123/datasets2/twoDoor/二门验收数据/20250310/datasets/test/";
  for (auto& dir : std::filesystem::directory_iterator(root)) {
    std::string path1 = dir.path().c_str();
    img11 = cv::imread(path1 + "/0.jpg");
    img21 = cv::imread(path1 + "/1.jpg");
    if (cnt == 0){
      cv::resize(img11, img11, cv::Size(256, 256));
      cv::resize(img21, img21, cv::Size(256, 256));
      cnt ++;
    } else if (cnt == 1) {
      cv::resize(img11, img11, cv::Size(244, 244));
      cv::resize(img21, img21, cv::Size(244, 244));
      cnt = 0;
    }
    images1.emplace_back(img11);
    images2.emplace_back(img21);
    if (images1.size() == 2) {
      auto start_t = std::chrono::steady_clock::now();
      Preprocess(images1, images2, device_input, stream);
      // 接下来可将 device_input 绑定到 TensorRT context 中进行推理
      auto pre_t = std::chrono::steady_clock::now();
      infer(context, bindings);
      auto infer_t = std::chrono::steady_clock::now();
      float result[batch_size];
      cudaMemcpy(result, output_d, batch_size * outputSize, cudaMemcpyDeviceToHost);
      printf("model cost time preprocess:%2.3lfms, infer:%2.3lfms\n",
            std::chrono::duration_cast<std::chrono::microseconds>(pre_t - start_t).count() / 1000.,
            std::chrono::duration_cast<std::chrono::microseconds>(infer_t - pre_t).count() / 1000.);
      double score = sigmoid(result[0]);
      double score2 = sigmoid(result[1]);
      std::cout << "Predicted score: " << score << ", "<< score2 << std::endl;
      if (score > 0.0) {
        cv::Mat combined1, combined2;
        cv::resize(images1[0], images1[0], cv::Size(224, 224));
        cv::resize(images2[0], images2[0], cv::Size(224, 224));
        cv::hconcat(images1[0], images2[0], combined1);
        cv::imshow("img1", combined1);
        cv::resize(images1[1], images1[1], cv::Size(224, 224));
        cv::resize(images2[1], images2[1], cv::Size(224, 224));
        cv::hconcat(images1[1], images2[1], combined2);
        cv::imshow("img2", combined2);
        cv::waitKey(0);
      }
      images1.clear();
      images2.clear();
    }
  }

  cudaFree(output_d);
  cudaFree(device_input);
  cudaStreamDestroy(stream);
}