/*
 * @Author: lihaobo
 * @Date: 2023-03-21 19:52:25
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-27 16:57:06
 * @Description: 请填写简介
 */
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"
#include "layer/abstract/layer.hpp"

// void BM_Func(benchmark::State& state) {
//      using namespace TinyTensor;
//     std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
//     input->index(0) = -1.f;
//     input->index(1) = -2.f;
//     input->index(2) = 3.f;
//     std::vector<std::shared_ptr<Tensor<float>>> inputs;
//     std::vector<std::shared_ptr<Tensor<float>>> outputs;
//     inputs.push_back(input);
//     outputs.push_back(input);
//     ReluLayer relu_layer;
//     relu_layer.Forward(inputs, outputs); }
// BENCHMARK(BM_Func)->Iterations(1000);


int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    
    google::InitGoogleLogging("MyTinyTensor");
    FLAGS_log_dir = "./log/";
    FLAGS_alsologtostderr = true;
   
     // 要注册所有 benchmark 测试      
    //benchmark_register_all_functions();
    // 注册benchmark测试      
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();


    LOG(INFO) << "Start test...\n";
    
    return RUN_ALL_TESTS();
}