#include "runtime/runtime_ir.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_initinoutput, init_init_input) {
  using namespace TinyTensor;
  const std::string &param_path = "../tmp/ten.pnnx.param";
  const std::string &bin_path = "../tmp/ten.pnnx.bin";
  RuntimeGraph graph(param_path, bin_path);
  graph.Init();
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const auto &operators = graph.operators();
  for (const auto &operator_ : operators) {
    LOG(INFO) << "type: " << operator_->type << " name: " << operator_->name;
    const std::map<std::string, std::shared_ptr<RuntimeOperand>>
        &input_operands_map = operator_->input_operands;
    for (const auto &input_operand : input_operands_map) {
      std::string shape_str;
      for (const auto &dim : input_operand.second->shapes) {
        shape_str += std::to_string(dim) + " ";
      }
      LOG(INFO) << "operand name: " << input_operand.first
                << " operand shape: " << shape_str;
    }
    LOG(INFO) << "-------------------------------------------------------";
  }
  graph.Build("pnnx_input_0", "pnnx_output_0");
  // 获取输入空间初始化后的Operator
  const auto &operators2 = graph.operators();
  const auto &operators3 = std::vector<std::shared_ptr<RuntimeOperator>>(
      operators2.begin() + 1, operators2.begin() + 5);

  // 获取name为conv1 relu1的2个op进行校验

  // 校验conv1
  // operand name: pnnx_input_0 operand shape: 2 3 128 128
  // 所以我们在下方要求size=2（也就是batch等于2） 通道c=3 rows = 128 cols = 128
  const auto &conv1 = *operators3.begin();
  const auto &conv1_input_operand = conv1->input_operands;
  ASSERT_EQ(conv1_input_operand.find("pnnx_input_0")->second->datas.size(), 2);
  const std::vector<std::shared_ptr<Tensor<float>>> &datas_conv1 =
      conv1_input_operand.at("pnnx_input_0")->datas; // datas是被准备好的空间
  for (const auto &data_conv1 : datas_conv1) {
    ASSERT_EQ(data_conv1->shapes().at(0), 3);
    ASSERT_EQ(data_conv1->shapes().at(1), 128);
    ASSERT_EQ(data_conv1->shapes().at(2), 128);
  }

  // operand name: conv1 operand shape: 2 64 128 128
  const auto &relu1 = *(operators3.begin() + 1);
  const auto &relu1_input_operand = relu1->input_operands;
  ASSERT_EQ(relu1_input_operand.find("conv1")->second->datas.size(), 2);
  const std::vector<std::shared_ptr<Tensor<float>>> &datas_relu1 =
      relu1_input_operand.at("conv1")->datas;
  for (const auto &data_relu1 : datas_relu1) {
    ASSERT_EQ(data_relu1->shapes().at(0), 64);
    ASSERT_EQ(data_relu1->shapes().at(1), 128);
    ASSERT_EQ(data_relu1->shapes().at(2), 128);
  }
}

TEST(test_initinoutput, init_init_graph) {
  using namespace TinyTensor;
  const std::string &param_path = "../tmp/test.pnnx.param";
  const std::string &bin_path = "../tmp/test.pnnx.bin";
  RuntimeGraph graph(param_path, bin_path);
  graph.Init();
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const auto &operators = graph.operators();
  for (const auto &op : operators) {
    LOG(INFO) << "type: " << op->type << " name: " << op->name;
  }
  // 按照这里的关系我们可以知道conv1后面是pnnx_expr_0
  // conv2 后面是Pnnx...
  // pnnx后面是max
  // max后面是output
  const auto &conv1 = *(operators.begin() + 1);
  ASSERT_EQ(conv1->name, "conv1");
  ASSERT_EQ(conv1->output_operators.size(), 1);
  const auto &conv1_output_ops = conv1->output_operators;
  ASSERT_EQ(conv1_output_ops.size(), 1);
  ASSERT_NE(conv1_output_ops.find("pnnx_expr_0"), conv1_output_ops.end());

  const auto &conv2 = *(operators.begin() + 2);
  ASSERT_EQ(conv2->name, "conv2");
  ASSERT_EQ(conv2->output_operators.size(), 1);
  const auto &conv2_output_ops = conv2->output_operators;
  ASSERT_EQ(conv2_output_ops.size(), 1);
  ASSERT_NE(conv2_output_ops.find("pnnx_expr_0"), conv1_output_ops.end());

  // 现在验证的是pnnx_expr层 它的输出节点为Max
  const auto &pnnx_expr = *(operators.begin() + 3);
  ASSERT_EQ(pnnx_expr->name, "pnnx_expr_0");
  ASSERT_EQ(pnnx_expr->output_operators.size(), 1);
  const auto &pnnx_expr_output_ops = pnnx_expr->output_operators;
  ASSERT_EQ(pnnx_expr_output_ops.size(), 1);
  ASSERT_NE(pnnx_expr_output_ops.find("max"), conv1_output_ops.end());
}

TEST(test_initinoutput, init_init_out) {
  using namespace TinyTensor;
  const std::string &param_path = "../tmp/ten.pnnx.param";
  const std::string &bin_path = "../tmp/ten.pnnx.bin";
  RuntimeGraph graph(param_path, bin_path);
  graph.Init();
  graph.Build("pnnx_input_0", "pnnx_output_0");

  const auto &operators = graph.operators();
  for (const auto &operator_ : operators) {
    LOG(INFO) << "type: " << operator_->type << " name: " << operator_->name;
    const auto &output_operands = operator_->output_operands;
    std::string shape_str;
    if (output_operands != nullptr) {
      for (const auto &output_shape : output_operands->shapes) {
        shape_str += std::to_string(output_shape) + " ";
      }
      LOG(INFO) << "operand name: " << output_operands->name
                << " operand shape: " << shape_str;
      LOG(INFO) << "-------------------------------------------------------";
    }
  }
  LOG(INFO) << "-------------------------------------------------------";

  // 开始检验输出空间有没有准备好？
  const auto &operators_sub = std::vector<std::shared_ptr<RuntimeOperator>>(
      operators.begin() + 1, operators.begin() + 3);
  ASSERT_EQ(operators_sub.size(), 2);
  const std::shared_ptr<RuntimeOperator> &conv1_operator =
      *operators_sub.begin();
  const std::shared_ptr<RuntimeOperand> &conv1_output =
      conv1_operator->output_operands;
  ASSERT_EQ(conv1_output->datas.size(), 2);
  for (const auto &conv_output_data : conv1_output->datas) {
    ASSERT_EQ(conv_output_data->channels(), 64);
    ASSERT_EQ(conv_output_data->rows(), 128);
    ASSERT_EQ(conv_output_data->cols(), 128);
  }

  // 检验relu

  const std::shared_ptr<RuntimeOperator> &relu_operator =
      *(operators_sub.begin() + 1);
  const std::shared_ptr<RuntimeOperand> &relu_output =
      relu_operator->output_operands;
  ASSERT_EQ(relu_output->datas.size(), 2);
  for (const auto &relu_output_data : relu_output->datas) {
    ASSERT_EQ(relu_output_data->channels(), 64);
    ASSERT_EQ(relu_output_data->rows(), 128);
    ASSERT_EQ(relu_output_data->cols(), 128);
  }
}
