#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "darknet.h"
#include "coco_names.h"
int main(int argc, char* argv[])
{
  std::cout << "hello\n";

  if (argc != 4)
  {
    std::cerr << "usage: yolov4 <cfg_path>, <weight_path> <image path>\n";
    return -1;
  }
  int input_image_size = 416;
   torch::DeviceType device_type;
   if(torch::cuda::is_available()){
      device_type = torch::kCUDA;
   }else{
     device_type = torch::kCPU;
   }
  torch::Device device(device_type);
  std::string cfg_file = argv[1];
  Darknet net(cfg_file.c_str(), &device);
  input_image_size = net.get_input_size();
  std::cout << "loading weight ..." << std::endl;
  net.load_darknet_weights(argv[2]);
  std::cout << "weight loaded ..." << std::endl;
  cv::Mat origin_image, resized_image;
  net.to(device);
  torch::NoGradGuard no_grad;
  net.eval();
  origin_image = cv::imread(argv[3]);
  cv::cvtColor(origin_image, resized_image, cv::COLOR_BGR2RGB);
  cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

  cv::Mat img_float;
  resized_image.convertTo(img_float, CV_32F, 1.0 / 255);

  auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3}).to(device);
  img_tensor = img_tensor.permute({0, 3, 1, 2});
  auto start = std::chrono::high_resolution_clock::now();
  auto result = net.predict(img_tensor, 80, 0.6, 0.4);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 
  std::cout<<  "time cost: " << duration.count()<< " ms\n";

  if (result.dim() == 1)
  {
    std::cout << "no object found" << std::endl;
  }
  else
  {
    int obj_num = result.size(0);

    std::cout << obj_num << " objects found" << std::endl;

    float w_scale = float(origin_image.cols) / input_image_size;
    float h_scale = float(origin_image.rows) / input_image_size;

    result.select(1, 1).mul_(w_scale);
    result.select(1, 2).mul_(h_scale);
    result.select(1, 3).mul_(w_scale);
    result.select(1, 4).mul_(h_scale);

    auto result_data = result.accessor<float, 2>();
 
    for (int i = 0; i < result.size(0); i++)
    {
      cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]), cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 0, 255), 1, 1, 0);
      int clas_id = static_cast<size_t>(result_data[i][7]);
      float score = result_data[i][6];
      std::string text = coco_class_names[clas_id] + "-" + std::to_string(score);
      cv::putText(origin_image,
                  text,
                  cv::Point(result_data[i][1] + 5, result_data[i][2] + 5),   // Coordinates
                  cv::FONT_HERSHEY_COMPLEX_SMALL,  // Font
                  1.0,                             // Scale. 2.0 = 2x bigger
                  cv::Scalar(255, 100, 255));      // BGR Color
    }
    cv::imwrite("det_result.jpg", origin_image);
  }
  std::cout << "Done" << std::endl;
}
