/*
 * @Author: lihaobo
 * @Date: 2023-03-06 10:07:41
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-07 10:16:55
 * @Description: 请填写简介
 */
#include<glog/logging.h>
#include "layer/relu_layer.hpp"
#include "ops/relu_op.hpp"
#include "factory/layer_factory.hpp"
namespace TinyTensor
{
    
ReluLayer::ReluLayer(const std::shared_ptr<ReluOperaor> &op){

}
    

void ReluLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                std::vector<std::shared_ptr<Tensor<float>>> &outputs)
{
                                    
                                
                                
                                
}



} // namespace TinyTensor
