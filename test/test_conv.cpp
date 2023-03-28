/*
 * @Author: lihaobo
 * @Date: 2023-03-27 16:46:38
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-28 13:30:38
 * @Description: 请填写简介
 */
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "ops/op.hpp"
#include "layer/conv_layer.hpp"

// 单卷积单通道
TEST(test_layer, conv1)
{
    using namespace TinyTensor;
    LOG(INFO) << "My convolution test!";
    ConvolutionOp *conv_op = new ConvolutionOp(false, 1, 1, 1, 0, 0);
    // 单个卷积核的情况
    std::vector<float> values;
    for (int i = 0; i < 3; ++i)
    {
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
    }
    std::shared_ptr<Tensor<float>> weight1 = std::make_shared<Tensor<float>>(1, 3, 3);
    weight1->Fill(values);
    //LOG(INFO) << "weight:";
    weight1->Show();
    // 设置权重
    std::vector<std::shared_ptr<Tensor<float>>> weights;
    weights.push_back(weight1);

    conv_op->set_weight(weights);
    std::shared_ptr<Operator> op = std::shared_ptr<ConvolutionOp>(conv_op);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    arma::fmat input_data = "1,2,3,4;"
                            "5,6,7,8;"
                            "7,8,9,10;"
                            "11,12,13,14";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 4, 4);
    input->at(0) = input_data;
    // LOG(INFO) << "input:";
    input->Show();
    // 权重数据和输入数据准备完毕
    inputs.push_back(input);
    
    ConvolutionLayer layer(op);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

    layer.Forwards(inputs, outputs);
    //LOG(INFO) << "result: ";
    for (int i = 0; i < outputs.size(); ++i)
    {
        outputs.at(i)->Show();
    }
}