/*
 * @Author: lihaobo
 * @Date: 2023-03-06 10:07:30
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-06 18:03:08
 * @Description: 请填写简介
 */
#include "layer/layer.hpp"
#include <glog/logging.h>

namespace TinyTensor{


Layer::Layer(const std::string &layer_name):_layer_name_(layer_name){

}
void Layer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                std::vector<std::shared_ptr<Tensor<float>>> &outputs){




}
}

