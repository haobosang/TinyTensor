#include "softmax.hpp"

namespace TinyTensor
{
SoftmaxLayer::SoftmaxLayer():Layer("SoftmaxLayer"){}

InferStatus SoftmaxLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs){

}
} // namespace TinyTensor
