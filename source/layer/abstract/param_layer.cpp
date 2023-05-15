/*
 * @Author: lihaobo
 * @Date: 2023-05-10 01:39:47
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-05-10 08:40:30
 * @Description: 
 */
#include "layer/abstract/param_layer.hpp"

#include <glog/logging.h>

namespace TinyTensor {
ParamLayer::ParamLayer(const std::string &layer_name) : Layer(layer_name) {}

void ParamLayer::InitBiasParam(const uint32_t param_count,
                               const uint32_t param_channel,
                               const uint32_t param_height,
                               const uint32_t param_width) {
  this->bias_ = std::vector<std::shared_ptr<Tensor<float>>>(param_count);
  for (uint32_t i = 0; i < param_count; ++i) {
    this->bias_.at(i) =
        std::make_shared<Tensor<float>>(param_channel, param_height, param_width);
  }
}

void ParamLayer::InitWeightParam(const uint32_t param_count,
                                 const uint32_t param_channel,
                                 const uint32_t param_height,
                                 const uint32_t param_width) {
  this->weights_ = std::vector<std::shared_ptr<Tensor<float>>>(param_count);
  for (uint32_t i = 0; i < param_count; ++i) {
    this->weights_.at(i) =
        std::make_shared<Tensor<float>>(param_channel, param_height, param_width);
  }
}

const std::vector<std::shared_ptr<Tensor<float>>> &ParamLayer::weights() const {
  return this->weights_;
}

const std::vector<std::shared_ptr<Tensor<float>>> &ParamLayer::bias() const {
  return this->bias_;
}

void ParamLayer::set_weights(
    const std::vector<std::shared_ptr<Tensor<float>>> &weights) {
  if (!this->bias_.empty()) {
    CHECK(weights.size() == weights_.size());
    for (uint32_t i = 0; i < weights.size(); ++i) {
      CHECK(this->weights_.at(i) != nullptr);
      CHECK(this->weights_.at(i)->rows() == weights.at(i)->rows());
      CHECK(this->weights_.at(i)->cols() == weights.at(i)->cols());
      CHECK(this->weights_.at(i)->channels() == weights.at(i)->channels());
    }
  }
  this->weights_ = weights;
}

void ParamLayer::set_bias(
    const std::vector<std::shared_ptr<Tensor<float>>> &bias) {
  if (!this->bias_.empty()) {
    CHECK(bias.size() == bias_.size());
    for (uint32_t i = 0; i < bias.size(); ++i) {
      CHECK(this->bias_.at(i) != nullptr);
      CHECK(this->bias_.at(i)->rows() == bias.at(i)->rows());
      CHECK(this->bias_.at(i)->cols() == bias.at(i)->cols());
      CHECK(this->bias_.at(i)->channels() == bias.at(i)->channels());
    }
  }
  this->bias_ = bias;
}

void ParamLayer::set_weights(const std::vector<float> &weights) {
  const uint32_t elem_size = weights.size();

  uint32_t weight_size = 0;
  const uint32_t batch_size = this->weights_.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    weight_size += this->weights_.at(i)->size();
  }

  CHECK_EQ(weight_size, elem_size);
  CHECK_EQ(elem_size % batch_size, 0);
  const uint32_t blob_size = elem_size / batch_size;
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    const uint32_t start_offset = idx * blob_size; // 当前卷积核参数的开始位置
    const uint32_t end_offset = start_offset + blob_size; // 当前卷积核参数的截至位置
    const auto &sub_values = std::vector<float>{weights.begin() + start_offset,
                                                weights.begin() + end_offset};
    this->weights_.at(idx)->Fill(sub_values);
  }
}

void ParamLayer::set_bias(const std::vector<float> &bias) {
  const uint32_t elem_size = bias.size();

  uint32_t bias_size = 0;
  const uint32_t batch_size = this->bias_.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    bias_size += this->bias_.at(i)->size();
  }

  CHECK_EQ(bias_size, elem_size);
  CHECK_EQ(elem_size % batch_size, 0);

  const uint32_t blob_size = elem_size / batch_size;
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    const uint32_t start_offset = idx * blob_size;
    const uint32_t end_offset = start_offset + blob_size;
    const auto &sub_values = std::vector<float>{bias.begin() + start_offset,
                                                bias.begin() + end_offset};
    this->bias_.at(idx)->Fill(sub_values);
  }
}

}  // namespace TinyTensor
