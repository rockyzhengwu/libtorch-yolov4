#include "darknet.h"
#include <string>
#include <fstream>
#include <iostream>
#include "config.h"
#include <stdio.h>
#include <vector>


struct Mish : torch::nn::Module
{
  Mish()
  {
    // torch::nn::Softplus softplus(torch::nn::SoftplusFuncOptions().beta(0.5).threshold(3.0));
    // register_module("softplus", softplus);
  }
  torch::Tensor forward(torch::Tensor x)
  {
    return x * torch::tanh(torch::nn::functional::softplus(x));
  }
};

struct MaxPoolLayer2D : torch::nn::Module
{
  int _kernel_size;
  int _stride;
  MaxPoolLayer2D(int kernel_size, int stride)
  {
    _kernel_size = kernel_size;
    _stride = stride;
  }

  torch::Tensor forward(torch::Tensor x)
  {
    if (_stride != 1)
    {
      x = torch::max_pool2d(x, {_kernel_size, _kernel_size}, {_stride, _stride});
    }
    else
    {
      int pad = _kernel_size - 1;
      torch::Tensor padded_x = torch::replication_pad2d(x, {0, pad, 0, pad});
      x = torch::max_pool2d(padded_x, {_kernel_size, _kernel_size}, {_stride, _stride});
    }

    return x;
  }
};

struct UpsampleLayer : torch::nn::Module
{
  int _stride;
  UpsampleLayer(int stride)
  {
    _stride = stride;
  }
  torch::Tensor forward(torch::Tensor x)
  {
    torch::IntArrayRef sizes = x.sizes();
    // torch::IntList sizes = x.sizes();
    int64_t w, h;
    if (sizes.size() == 4)
    {
      w = sizes[2] * _stride;
      h = sizes[3] * _stride;
      x = torch::upsample_nearest2d(x, {w, h});
    }
    else if (sizes.size() == 3)
    {
      w = sizes[2] * _stride;
      x = torch::upsample_nearest1d(x, {w});
    }
    return x;
  }
};

struct EmptyLayer : torch::nn::Module
{
  EmptyLayer()
  {
  }

  torch::Tensor forward(torch::Tensor x)
  {
    return x;
  }
};

struct DetectionLayer : torch::nn::Module
{
  std::vector<float> _anchors;

  DetectionLayer(std::vector<float> anchors)
  {
    _anchors = anchors;
  }

  torch::Tensor forward(torch::Tensor prediction, int inp_dim, int num_classes, torch::Device device)
  {
    return predict_transform(prediction, inp_dim, _anchors, num_classes, device);
  }

  torch::Tensor predict_transform(torch::Tensor prediction, int inp_dim, std::vector<float> anchors, int num_classes, torch::Device device)
  {
    int batch_size = prediction.size(0);
    int stride = floor(inp_dim / prediction.size(2));
    int grid_size = floor(inp_dim / stride);
    int bbox_attrs = 5 + num_classes;
    int num_anchors = anchors.size() / 2;
    unsigned int i;

    for (i = 0; i < anchors.size(); i++)
    {
      anchors[i] = anchors[i] / stride;
    }
    torch::Tensor result = prediction.view({batch_size, bbox_attrs * num_anchors, grid_size * grid_size});
    result = result.transpose(1, 2).contiguous();
    result = result.view({batch_size, grid_size * grid_size * num_anchors, bbox_attrs});

    result.select(2, 0).sigmoid_();
    result.select(2, 1).sigmoid_();
    result.select(2, 4).sigmoid_();

    auto grid_len = torch::arange(grid_size);

    std::vector<torch::Tensor> args = torch::meshgrid({grid_len, grid_len});

    torch::Tensor x_offset = args[1].contiguous().view({-1, 1});
    torch::Tensor y_offset = args[0].contiguous().view({-1, 1});


    x_offset = x_offset.to(device);
    y_offset = y_offset.to(device);

    auto x_y_offset = torch::cat({x_offset, y_offset}, 1).repeat({1, num_anchors}).view({-1, 2}).unsqueeze(0);
    result.slice(2, 0, 2).add_(x_y_offset);

    torch::Tensor anchors_tensor = torch::from_blob(anchors.data(), {num_anchors, 2});
    //if (device != nullptr)
    anchors_tensor = anchors_tensor.to(device);
    anchors_tensor = anchors_tensor.repeat({grid_size * grid_size, 1}).unsqueeze(0);

    result.slice(2, 2, 4).exp_().mul_(anchors_tensor);
    result.slice(2, 5, 5 + num_classes).sigmoid_();
    result.slice(2, 0, 4).mul_(stride);

    return result;
  }
};

Darknet::Darknet(const char *conf_file, torch::Device *device)
{
  config_ = Config(conf_file);
  device_ = device;
  create_modules();
}

void Darknet::create_modules()
{
  size_t i;
  Block block;
  std::string block_type;

  std::vector<int> out_channels;
  std::vector<int> out_widths;
  std::vector<int> out_heights;

  size_t index = 0;

  printf("layer       filters    size              input                output\n");
  for (i = 0; i < config_.blocks_.size(); i++)
  {
    torch::nn::Sequential module;
    block = config_.blocks_.at(i);
    block_type = block["type"];
    if (block_type == "net")
    {
      out_channels.push_back(Config::get_int_from_block(block, "channels", 3));
      out_widths.push_back( Config::get_int_from_block(block, "width", 0));
      out_heights.push_back(Config::get_int_from_block(block, "height", 0));
      continue;
    }
    else if (block_type == "convolutional")
    {
      std::string activation = Config::get_string_from_block(block, "activation", "");
      int batch_normalize = Config::get_int_from_block(block, "batch_normalize", 0);
      int filters = Config::get_int_from_block(block, "filters", 0);
      int is_pad = Config::get_int_from_block(block, "pad", 0);
      int kernel_size = Config::get_int_from_block(block, "size", 0);
      int stride = Config::get_int_from_block(block, "stride", 1);

      int pad = is_pad > 0 ? (kernel_size - 1) / 2 : 0;
      bool with_bias = batch_normalize > 0 ? false : true;
      torch::nn::Conv2dOptions conv_options(out_channels.back(), filters, kernel_size);
      conv_options.stride(stride);
      conv_options.padding(pad);
      conv_options.groups(1);
      conv_options.bias(with_bias);
      torch::nn::Conv2d conv(conv_options);
      module->push_back(conv);
      int width = (out_widths.back() + 2 * pad - kernel_size )/stride + 1;
      int height = (out_heights.back() +2 * pad - kernel_size )/stride + 1;
      printf("%5zu %-6s %4d    %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d\n",
                 i, "conv", filters, kernel_size, kernel_size, stride,
                 out_widths.back(), out_heights.back(), out_channels.back(), width, height, filters);
      out_heights.push_back(height);
      out_widths.push_back(width);
      out_channels.push_back(filters);
      if (batch_normalize > 0)
      {
        torch::nn::BatchNormOptions bn_options = torch::nn::BatchNormOptions(filters);
        bn_options.affine(true);
        bn_options.track_running_stats(true);
        torch::nn::BatchNorm2d bn = torch::nn::BatchNorm2d(bn_options);
        module->push_back(bn);
      }
      if (activation == "leaky")
      {
        module->push_back(torch::nn::Functional(torch::leaky_relu, /*slope=*/0.1));
      }
      else if (activation == "mish")
      {
        Mish mish;
        module->push_back(mish);
      }
      else if (activation == "linear")
      {
      }
      else
      {
        std::cout << "not supperted activation: " << activation << "\n";
      }
    }
    else if (block_type == "upsample")
    {
      int stride = Config::get_int_from_block(block, "stride", 1);
      int32_t width = out_widths.back() * stride;
      int32_t height = out_heights.back() * stride;
      UpsampleLayer uplayer(stride);
      printf("%5zu %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d \n" ,
                i, "upsample", stride, out_widths.back(), out_heights.back(), out_channels.back(), width, height, out_channels.back());
      module->push_back(uplayer);
      out_widths.push_back(width);
      out_heights.push_back(height);
      out_channels.push_back(out_channels.back());
    }
    else if (block_type == "maxpool")
    {
      int stride = Config::get_int_from_block(block, "stride", 1);
      int size = Config::get_int_from_block(block, "size", 1);
      MaxPoolLayer2D poolLayer(size, stride);
      module->push_back(poolLayer);

      int width = out_widths.back()/ stride;
      int height = out_heights.back() / stride;
      printf("%5zu %-6s         %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d\n",
             i, "max", size, size, stride, 
             out_widths.back(), out_heights.back(), out_channels.back(), width, height,out_channels.back());

      out_widths.push_back(width);
      out_heights.push_back(height);
      out_channels.push_back(out_channels.back());
    }
    else if (block_type == "shortcut")
    {
      int from = Config::get_int_from_block(block, "from", 0);
      printf("%5zu %-6s %d \n", i, "shortcut", from);
      block["from"] = std::to_string(from);
      config_.blocks_[i] = block;
      EmptyLayer layer;
      module->push_back(layer);
      out_channels.push_back(out_channels.back());
      out_heights.push_back(out_heights.back());
      out_widths.push_back(out_widths.back());
    }

    else if (block_type == "route")
    {
      // L 85: -1, 61
      std::string layers_info = Config::get_string_from_block(block, "layers", "");
      std::vector<int> layers;
      int groups = Config::get_int_from_block(block, "groups", 0);
      // size_t group_id = Config::get_int_from_block(block, "group_id", 0);
      Config::split(layers_info, layers, ",");
      int32_t total_channel = 0;
      for(size_t j=0; j< layers.size(); j++){
        int ix = layers[j] > 0 ? layers[j]+1 : (i + layers[j]);
        total_channel += out_channels[ ix ];
        layers[j] = ix;
      }
      if (layers.size()==4){
        printf("%5zu %-6s %d %d %d %d                              %d\n",
         i, "route",  layers[0], layers[1], layers[2], layers[3], total_channel);
      }else if(layers.size()==2){
        printf("%5zu %-6s %d %d                                    %d\n",
         i, "route", layers[0], layers[1], total_channel);
      }else if(layers.size()==1){
        if (groups!=0){
          total_channel = total_channel / groups;
          block["chunk_size"] = std::to_string(total_channel);
        }
        printf("%5zu %-6s %d                                       %d\n",  
        i, "route",  layers[0], total_channel);
      } else {
        printf("not suppert route \n");
      }

      out_channels.push_back(total_channel);
      out_widths.push_back(out_widths.back());
      out_heights.push_back(out_widths.back());
      config_.blocks_[i] = block;
      // placeholder
      EmptyLayer layer;
      module->push_back(layer);
    }
    else if (block_type == "yolo")
    {
      std::string mask_info = Config::get_string_from_block(block, "mask", "");
      std::vector<int> masks;
      Config::split(mask_info, masks, ",");

      std::string anchor_info = Config::get_string_from_block(block, "anchors", "");
      std::vector<int> anchors;
      Config::split(anchor_info, anchors, ",");
      std::vector<float> anchor_points;
      printf("%5zu %-6s   \n", i, "yolo" );
      int pos;
      size_t num= Config::get_int_from_block(block, "num", 0);
      size_t stride = anchors.size() / num;

      for (size_t j = 0; j < masks.size(); j++)
      {
        pos = masks[j];
        anchor_points.push_back(anchors[pos * stride]);
        anchor_points.push_back(anchors[pos * stride + 1]);
      }
      DetectionLayer layer(anchor_points);
      module->push_back(layer);
      out_channels.push_back(out_channels.back());
      out_widths.push_back(out_widths.back());
      out_heights.push_back(out_heights.back());
    }
    else
    {
      std::cout << "unsupported operator:" << block["type"] << std::endl;
    }
    module_list.push_back(module);
    char *module_key = new char[strlen("layer_") + sizeof(index) + 1];
    sprintf(module_key, "%s%zu", "layer_", index);
    register_module(module_key, module);
    index += 1;
    assert(out_widths.size() == index+1);
    assert(out_heights.size()==index+1);
    assert(out_channels.size() == index+1);
  }
}

void Darknet::create_convolutional(torch::nn::Sequential &module, Block &block, int in_channels)
{

}

void Darknet::load_darknet_weights(const char *weights_file)
{
  std::ifstream fs(weights_file, std::ios::binary);
  // header info: 5 * int32_t
  int32_t header_size = sizeof(int32_t) * 5;
  int64_t index_weight = 0;

  fs.seekg(0, fs.end);
  int64_t length = fs.tellg();
  // skip header
  length = length - header_size;

  fs.seekg(header_size, fs.beg);
  float *weights_src = (float *)malloc(length);
  fs.read(reinterpret_cast<char *>(weights_src), length);
  fs.close();

  // at::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
  at::Tensor weights = torch::from_blob(weights_src, {length / 4});
  for (size_t i = 0; i < module_list.size(); i++)
  {
    Block module_info = config_.blocks_[i + 1];

    std::string module_type = module_info["type"];

    // only conv layer need to load weight
    if (module_type != "convolutional")
      continue;
    torch::nn::Sequential seq_module = module_list[i];

    auto conv_module = seq_module.ptr()->ptr(0);
    torch::nn::Conv2dImpl *conv_imp = dynamic_cast<torch::nn::Conv2dImpl *>(conv_module.get());

    int batch_normalize = Config::get_int_from_block(module_info, "batch_normalize", 0);
    if (batch_normalize > 0)
    {
      // second module
      auto bn_module = seq_module.ptr()->ptr(1);

      torch::nn::BatchNorm2dImpl *bn_imp = dynamic_cast<torch::nn::BatchNorm2dImpl *>(bn_module.get());

      int num_bn_biases = bn_imp->bias.numel();

      at::Tensor bn_bias = weights.slice(0, index_weight, index_weight + num_bn_biases);
      index_weight += num_bn_biases;

      at::Tensor bn_weights = weights.slice(0, index_weight, index_weight + num_bn_biases);
      index_weight += num_bn_biases;

      at::Tensor bn_running_mean = weights.slice(0, index_weight, index_weight + num_bn_biases);
      index_weight += num_bn_biases;

      at::Tensor bn_running_var = weights.slice(0, index_weight, index_weight + num_bn_biases);
      index_weight += num_bn_biases;
      bn_bias = bn_bias.view_as(bn_imp->bias);
      bn_weights = bn_weights.view_as(bn_imp->weight);
      bn_running_mean = bn_running_mean.view_as(bn_imp->running_mean);
      bn_running_var = bn_running_var.view_as(bn_imp->running_var);

      bn_imp->bias.set_data(bn_bias);
      bn_imp->weight.set_data(bn_weights);
      bn_imp->running_mean.set_data(bn_running_mean);
      bn_imp->running_var.set_data(bn_running_var);
    }
    else
    {
      int num_conv_biases = conv_imp->bias.numel();
      at::Tensor conv_bias = weights.slice(0, index_weight, index_weight + num_conv_biases);
      index_weight += num_conv_biases;

      conv_bias = conv_bias.view_as(conv_imp->bias);
      conv_imp->bias.set_data(conv_bias);
    }
    int num_weights = conv_imp->weight.numel();
    at::Tensor conv_weights = weights.slice(0, index_weight, index_weight + num_weights);
    index_weight += num_weights;
    conv_weights = conv_weights.view_as(conv_imp->weight);
    conv_imp->weight.set_data(conv_weights);
  }
}

// returns the IoU of two bounding boxes
static inline torch::Tensor get_bbox_iou(torch::Tensor box1, torch::Tensor box2)
{
  // Get the coordinates of bounding boxes
  torch::Tensor b1_x1, b1_y1, b1_x2, b1_y2;
  b1_x1 = box1.select(1, 0);
  b1_y1 = box1.select(1, 1);
  b1_x2 = box1.select(1, 2);
  b1_y2 = box1.select(1, 3);
  torch::Tensor b2_x1, b2_y1, b2_x2, b2_y2;
  b2_x1 = box2.select(1, 0);
  b2_y1 = box2.select(1, 1);
  b2_x2 = box2.select(1, 2);
  b2_y2 = box2.select(1, 3);

  // et the corrdinates of the intersection rectangle
  torch::Tensor inter_rect_x1 = torch::max(b1_x1, b2_x1);
  torch::Tensor inter_rect_y1 = torch::max(b1_y1, b2_y1);
  torch::Tensor inter_rect_x2 = torch::min(b1_x2, b2_x2);
  torch::Tensor inter_rect_y2 = torch::min(b1_y2, b2_y2);

  // Intersection area
  torch::Tensor inter_area = torch::max(inter_rect_x2 - inter_rect_x1 + 1, torch::zeros(inter_rect_x2.sizes())) * torch::max(inter_rect_y2 - inter_rect_y1 + 1, torch::zeros(inter_rect_x2.sizes()));

  // Union Area
  torch::Tensor b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1);
  torch::Tensor b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1);

  torch::Tensor iou = inter_area / (b1_area + b2_area - inter_area);

  return iou;
}

torch::Tensor Darknet::predict(torch::Tensor input, int num_classes, float confidence, float nms_conf)
{
  auto output = this->forward(input);
  auto result = nms(output, num_classes, confidence, nms_conf);
  return result;
}

torch::Tensor Darknet::nms(torch::Tensor prediction, int num_classes, float confidence, float nms_conf)
{
  // get result which object confidence > threshold
  auto conf_mask = (prediction.select(2, 4) > confidence).to(torch::kFloat32).unsqueeze(2);

  prediction.mul_(conf_mask);
  auto ind_nz = torch::nonzero(prediction.select(2, 4)).transpose(0, 1).contiguous();

  if (ind_nz.size(0) == 0)
  {
    return torch::zeros({0});
  }

  torch::Tensor box_a = torch::ones(prediction.sizes(), prediction.options());
  // top left x = centerX - w/2
  box_a.select(2, 0) = prediction.select(2, 0) - prediction.select(2, 2).div(2);
  box_a.select(2, 1) = prediction.select(2, 1) - prediction.select(2, 3).div(2);
  box_a.select(2, 2) = prediction.select(2, 0) + prediction.select(2, 2).div(2);
  box_a.select(2, 3) = prediction.select(2, 1) + prediction.select(2, 3).div(2);

  prediction.slice(2, 0, 4) = box_a.slice(2, 0, 4);

  int batch_size = prediction.size(0);
  int item_attr_size = 5;

  torch::Tensor output = torch::ones({1, prediction.size(2) + 1});
  bool write = false;

  int num = 0;

  for (int i = 0; i < batch_size; i++)
  {
    auto image_prediction = prediction[i];

    // get the max classes score at each result
    std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(image_prediction.slice(1, item_attr_size, item_attr_size + num_classes), 1);

    // class score
    auto max_conf = std::get<0>(max_classes);
    // index
    auto max_conf_score = std::get<1>(max_classes);
    max_conf = max_conf.to(torch::kFloat32).unsqueeze(1);
    max_conf_score = max_conf_score.to(torch::kFloat32).unsqueeze(1);

    // shape: n * 7, left x, left y, right x, right y, object confidence, class_score, class_id
    image_prediction = torch::cat({image_prediction.slice(1, 0, 5), max_conf, max_conf_score}, 1);

    // remove item which object confidence == 0
    auto non_zero_index = torch::nonzero(image_prediction.select(1, 4));
    auto image_prediction_data = image_prediction.index_select(0, non_zero_index.squeeze()).view({-1, 7});

    // get unique classes
    std::vector<torch::Tensor> img_classes;

    for (size_t m = 0, len = image_prediction_data.size(0); m < len; m++)
    {
      bool found = false;
      for (size_t n = 0; n < img_classes.size(); n++)
      {
        auto ret = (image_prediction_data[m][6] == img_classes[n]);
        if (torch::nonzero(ret).size(0) > 0)
        {
          found = true;
          break;
        }
      }
      if (!found)
        img_classes.push_back(image_prediction_data[m][6]);
    }

    for (size_t k = 0; k < img_classes.size(); k++)
    {
      auto cls = img_classes[k];

      auto cls_mask = image_prediction_data * (image_prediction_data.select(1, 6) == cls).to(torch::kFloat32).unsqueeze(1);
      auto class_mask_index = torch::nonzero(cls_mask.select(1, 5)).squeeze();

      auto image_pred_class = image_prediction_data.index_select(0, class_mask_index).view({-1, 7});
      // ascend by confidence
      // seems that inverse method not work
      std::tuple<torch::Tensor, torch::Tensor> sort_ret = torch::sort(image_pred_class.select(1, 4));

      auto conf_sort_index = std::get<1>(sort_ret);

      // seems that there is something wrong with inverse method
      // conf_sort_index = conf_sort_index.inverse();

      image_pred_class = image_pred_class.index_select(0, conf_sort_index.squeeze()).cpu();

      for (int w = 0; w < image_pred_class.size(0) - 1; w++)
      {
        int mi = image_pred_class.size(0) - 1 - w;

        if (mi <= 0)
        {
          break;
        }

        auto ious = get_bbox_iou(image_pred_class[mi].unsqueeze(0), image_pred_class.slice(0, 0, mi));

        auto iou_mask = (ious < nms_conf).to(torch::kFloat32).unsqueeze(1);
        image_pred_class.slice(0, 0, mi) = image_pred_class.slice(0, 0, mi) * iou_mask;

        // remove from list
        auto non_zero_index = torch::nonzero(image_pred_class.select(1, 4)).squeeze();
        image_pred_class = image_pred_class.index_select(0, non_zero_index).view({-1, 7});
      }

      torch::Tensor batch_index = torch::ones({image_pred_class.size(0), 1}).fill_(i);

      if (!write)
      {
        output = torch::cat({batch_index, image_pred_class}, 1);
        write = true;
      }
      else
      {
        auto out = torch::cat({batch_index, image_pred_class}, 1);
        output = torch::cat({output, out}, 0);
      }
      num += 1;
    }
  }

  if (num == 0)
  {
    return torch::zeros({0});
  }

  return output;
}

torch::Tensor Darknet::forward(torch::Tensor x)
{

  int module_count = module_list.size();

  std::vector<torch::Tensor> outputs(module_count);

  torch::Tensor result;
  int write = 0;

  for (int i = 0; i < module_count; i++)
  {
    Block block = config_.blocks_[i + 1];
    std::string layer_type = block["type"];

    if (layer_type == "net")
      continue;

    if (layer_type == "convolutional" || layer_type == "upsample" || layer_type == "maxpool")
    {
      torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *>(module_list[i].ptr().get());
      x = seq_imp->forward(x);
      outputs[i] = x;
    }
    else if (layer_type == "route")
    {
      std::vector<int> layers;
      int groups = Config::get_int_from_block(block, "groups", 0);
      size_t group_id = Config::get_int_from_block(block, "group_id", 0);
      Config::split(block["layers"], layers, ",");
      for(size_t j =0; j< layers.size(); j++){
        layers[j] = layers[j]  > 0 ? layers[j] : layers[j] + i;
      }
      if(layers.size()==1){
        x = outputs[layers[0]];
        if (groups!=0){
          int64_t chunk_size = Config::get_int_from_block(block, "chunk_size", 0);
          std::vector<torch::Tensor> group_tensor = at::split(x, chunk_size, 1);
          x = group_tensor.at(group_id);
        }
      }else if(layers.size()==2){
        torch::Tensor map_1 = outputs[layers[0]];
        torch::Tensor map_2 = outputs[layers[1]];
        x = torch::cat({map_1, map_2},1);
      }else if(layers.size()==4){
        torch::Tensor map_1 = outputs[layers[0]];
        torch::Tensor map_2 = outputs[layers[1]];
        torch::Tensor map_3 = outputs[layers[2]];
        torch::Tensor map_4 = outputs[layers[3]];
        x = torch::cat({map_1, map_2, map_3, map_4},1);
      }
      outputs[i]=x;
    }
    else if (layer_type == "shortcut")
    {
      int from = std::stoi(block["from"]);
      from = from > 0 ? from : from + i;
      x = outputs[i - 1] + outputs[from];
      outputs[i] = x;
    }
    else if (layer_type == "yolo")
    {
      torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *>(module_list[i].ptr().get());
      Block net_info = config_.blocks_[0];
      int inp_dim = Config::get_int_from_block(net_info, "height", 0);
      int num_classes = Config::get_int_from_block(block, "classes", 0);
      x = seq_imp->forward(x, inp_dim, num_classes, *device_);
      if (write == 0)
      {
        result = x;
        write = 1;
      }
      else
      {
        result = torch::cat({result, x}, 1);
      }
      outputs[i] = outputs[i - 1];
    }else{
      std::cout << "unknown type " << block["type"] << "\n";
    }
  }
  return result;
}

void Darknet::show_config()
{
  size_t i;
  for (i = 0; i < config_.blocks_.size(); i++)
  {
    std::cout << config_.blocks_[i] << std::endl;
  }
}

int Darknet::get_input_size(){
  return Config::get_int_from_block(config_.blocks_[0], "height", 0);
}