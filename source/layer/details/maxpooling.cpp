#include "layer/details/maxpooling.hpp"
namespace TinyTensor
{
MaxPoolingLayer::MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w, uint32_t pooling_size_h,
                  uint32_t pooling_size_w, uint32_t stride_h, uint32_t stride_w):Layer("MaxPooling"),padding_h_(padding_h),padding_w_(padding_w),
                  pooling_h_(pooling_size_h),pooling_w_(pooling_size_w),stride_h_(stride_h),stride_w_(stride_w){}
InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs){

}

static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                              std::shared_ptr<Layer> &max_layer){


}

} // namespace TinyTensor
