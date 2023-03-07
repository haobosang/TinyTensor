/*
 * @Author: lihaobo
 * @Date: 2023-03-06 10:07:41
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-07 20:33:03
 * @Description: 请填写简介
 */
#include <glog/logging.h>
#include "ops/relu_op.hpp"
#include "layer/relu_layer.hpp"
#include "factory/layer_factory.hpp"

namespace TinyTensor
{
    
ReluLayer::ReluLayer(const std::shared_ptr<Operator> &op):Layer("Relu"){
    CHECK(op->op_type_ == OpType::kOperatorRelu);

    ReluOperaor *relu_op = dynamic_cast<ReluOperaor *>(op.get());

    CHECK(relu_op != nullptr);

    this->_op = std::make_shared<ReluOperaor>(relu_op->get_thresh());

}
    

void ReluLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                std::vector<std::shared_ptr<Tensor<float>>> &outputs)
{
    CHECK(this->_op !=nullptr);
    CHECK(this->_op->op_type_ == OpType::kOperatorRelu);

    const uint32_t batch_size = inputs.size();


    for(uint32_t i =0;i<batch_size;i++)
    {
        CHECK(!inputs.at(i)->empty());
        const std::shared_ptr<Tensor<float>> &inputs_data = inputs.at(i);

        inputs_data->data().transform([&](float value){
            float thresh = _op->get_thresh();
            if(value>=thresh)
            {
                return value;
            }else{
                return 0.f;
            }

        });

        outputs.push_back(inputs_data);


    }

                                
                                
}



} // namespace TinyTensor
