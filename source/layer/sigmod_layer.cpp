/*
 * @Author: lihaobo
 * @Date: 2023-03-10 10:13:29
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-10 13:12:00
 * @Description: 请填写简介
 */
#include "ops/sigmod_op.hpp"
#include "layer/sigmod_layer.hpp"
#include "glog/logging.h"
#include "factory/layer_factory.hpp"
#include <cmath>

namespace TinyTensor{

SigmodLayer::SigmodLayer(const std::shared_ptr<Operator> &op):Layer("Sigmod"){
    CHECK(op->op_type_ == OpType::kOperatorSigmod);

    SigmodOperaor *sigmod_op = dynamic_cast<SigmodOperaor *>(op.get());

    CHECK(sigmod_op != nullptr);

    this->_op = std::make_shared<SigmodOperaor>();


}


void SigmodLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                std::vector<std::shared_ptr<Tensor<float>>> &outputs){

    CHECK(this->_op !=nullptr);
    CHECK(this->_op->op_type_ == OpType::kOperatorSigmod);

    const uint32_t batch_size = inputs.size();

    for(uint32_t i = 0;i< batch_size;i++)
    {
        CHECK(!inputs.at(i)->empty());
        const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
        std::shared_ptr<Tensor<float>> output_data = input_data->Clone();

        output_data ->data().transform([&](float value){
            //float y = - value;
            return 1.0/(1 + exp(- value));
        });
        outputs.push_back(output_data);
        

    }

}

std::shared_ptr<Layer> SigmodLayer::CreateInstance(const std::shared_ptr<Operator> &op){
    std::shared_ptr<Layer> sigmod_layer = std::make_shared<SigmodLayer>(op);
    return sigmod_layer;
    
}
LayerRegistererWrapper KSigmodLayer(OpType::kOperatorSigmod, SigmodLayer::CreateInstance);    
}