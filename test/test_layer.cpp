#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/layer/details/relu.hpp"
#include "runtime/runtime_ir.hpp"
#include "layer/abstract/layer.hpp"
#include <benchmark/benchmark.h>
TEST(test_layer, forward_relu) {
    using namespace TinyTensor;
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 3.f;
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    inputs.push_back(input);
    outputs.push_back(input);
    ReluLayer relu_layer;
    relu_layer.Forward(inputs, outputs);

    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
  
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
    }
}

