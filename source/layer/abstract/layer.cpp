#include "layer/abstract/layer.hpp"
namespace TinyTensor {

const std::vector<std::shared_ptr<Tensor<float>>> &Layer::weights() const {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

const std::vector<std::shared_ptr<Tensor<float>>> &Layer::bias() const {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_bias(const std::vector<float> &bias) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &bias) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_weights(const std::vector<float> &weights) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_weights(const std::vector<std::shared_ptr<Tensor<float>>> &weights) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}


InferStatus Layer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                           std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

}