#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/layer/details/relu.hpp"
#include "../source/layer/details/sigmoid.hpp"
#include "runtime/runtime_ir.hpp"
#include "layer/abstract/layer.hpp"
// TEST(test_layer, forward_relu) {
//     using namespace TinyTensor;
//     std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
//     input->index(0) = -1.f;
//     input->index(1) = -2.f;
//     input->index(2) = 3.f;
//     std::vector<std::shared_ptr<Tensor<float>>> inputs;
//     std::vector<std::shared_ptr<Tensor<float>>> outputs;
//     inputs.push_back(input);
//     outputs.push_back(input);
//     ReluLayer relu_layer;
//     relu_layer.Forward(inputs, outputs);

//     ASSERT_EQ(outputs.size(), 1);
//     for (int i = 0; i < outputs.size(); ++i) {
  
//     ASSERT_EQ(outputs.at(i)->index(0), 0.f);
//     ASSERT_EQ(outputs.at(i)->index(1), 0.f);
//     ASSERT_EQ(outputs.at(i)->index(2), 3.f);
//     }
// }
TEST(test_layer, forward_sigmoid) {
    using namespace TinyTensor;
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 3.f;
    std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(input->shapes());

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    inputs.push_back(input);
    outputs.push_back(output);

    SigmoidLayer sigmoid;
    sigmoid.Forward(inputs, outputs);

    // ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->index(0), 1 / (1 + std::exp(-inputs.at(i)->index(0))));
        ASSERT_EQ(outputs.at(i)->index(1), 1 / (1 + std::exp(-inputs.at(i)->index(1))));
        ASSERT_EQ(outputs.at(i)->index(2), 1 / (1 + std::exp(-inputs.at(i)->index(2))));
    }

}

