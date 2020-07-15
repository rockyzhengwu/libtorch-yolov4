#ifndef DARKNET_H_
#define DARKNET_H_

#include <string>
#include <vector>
#include <map>
#include <torch/torch.h>

#include "config.h"


struct Darknet: torch::nn::Module {
  public:
  	Darknet(const char *conf_file, torch::Device *device);
    void create_modules();

  	torch::Tensor forward(torch::Tensor x);
    void load_darknet_weights(const char *weights_file);
    torch::Tensor predict(torch::Tensor input, int num_classes, float confidence, float nms_conf=0.4);
    torch::Tensor nms(torch::Tensor input, int num_classes, float confidence, float nms_conf=0.4);
    void show_config();
    int get_input_size();



  private:
  	torch::Device *device_;
    void create_convolutional(torch::nn::Sequential &module, Block &block, int in_channels);
    Config config_;
	  std::vector<torch::nn::Sequential> module_list;

    // void create_modules();
};
#endif