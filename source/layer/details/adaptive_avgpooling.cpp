#include "layer/details/adaptive_avgpooling.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
namespace  TinyTensor
{
AdaptiveAveragePoolingLayer::AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w):Layer(""),output_h_(output_h),
                                                                    output_w_(output_w){}

InferStatus AdaptiveAveragePoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs){

                }

ParseParameterAttrStatus AdaptiveAveragePoolingLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                        std::shared_ptr<Layer> &avg_layer){

                                        }
    
} // namespace  TinyTensor

