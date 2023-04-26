
#include "runtime/runtime_ir.hpp"
#include <memory>
#include <iostream>
#include <iomanip>
#include <queue>
#include <deque>
#include <utility>
#include "factory/layer_factory.hpp"

namespace TinyTensor {

void RuntimeGraphShape::InitOperatorInputTensor(
    const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {
  if (operators.empty()) {
    LOG(ERROR) << "Operators for init input shapes is empty!";
    return;
  }
  for (const auto &op : operators) {
    if (op->input_operands.empty()) {
      continue;
    } else {
      const std::map<std::string, std::shared_ptr<RuntimeOperand>> &
          input_operands_map = op->input_operands;
      for (const auto &input_operand_iter : input_operands_map) {
        const auto &input_operand = input_operand_iter.second;
        const auto &type = input_operand->type;
        CHECK(type == RuntimeDataType::kTypeFloat32)
                << "The graph only support float32 yet!";
        const auto &input_operand_shape = input_operand->shapes;
        auto &input_datas = input_operand->datas;

        CHECK(!input_operand_shape.empty());
        const int32_t batch = input_operand_shape.at(0);
        CHECK(batch >= 0) << "Dynamic batch size is not supported!";
        CHECK(input_operand_shape.size() == 2 ||
            input_operand_shape.size() == 4 ||
            input_operand_shape.size() == 3)
                << "Unsupported tensor shape sizes: " << input_operand_shape.size();

        if (!input_datas.empty()) {
          CHECK(input_datas.size() == batch) << "Batch size is wrong!";
          for (int32_t i = 0; i < batch; ++i) {
            const std::vector<uint32_t> &input_data_shape =
                input_datas.at(i)->shapes();
            CHECK(input_data_shape.size() == 3)
                    << "THe origin shape size of operator input data do not equals "
                       "to three";
            if (input_operand_shape.size() == 4) {
              CHECK(input_data_shape.at(0) == input_operand_shape.at(1) &&
                  input_data_shape.at(1) == input_operand_shape.at(2) &&
                  input_data_shape.at(2) == input_operand_shape.at(3));
            } else if (input_operand_shape.size() == 2) {
              CHECK(input_data_shape.at(1) == input_operand_shape.at(1) &&
                  input_data_shape.at(0) == 1 && input_data_shape.at(2) == 1);
            } else {
              // current shape size = 3
              CHECK(input_data_shape.at(1) == input_operand_shape.at(1) &&
                  input_data_shape.at(0) == 1 &&
                  input_data_shape.at(2) == input_operand_shape.at(2));
            }
          }
        } else {
          input_datas.resize(batch);
          for (int32_t i = 0; i < batch; ++i) {
            if (input_operand_shape.size() == 4) {
              input_datas.at(i) = std::make_shared<Tensor<float>>(
                  input_operand_shape.at(1), input_operand_shape.at(2),
                  input_operand_shape.at(3));
            } else if (input_operand_shape.size() == 2) {
              input_datas.at(i) = std::make_shared<Tensor<float>>(
                  1, input_operand_shape.at(1), 1);
            } else {
              // current shape is 3
              input_datas.at(i) = std::make_shared<Tensor<float>>(
                  1, input_operand_shape.at(1), input_operand_shape.at(2));
            }
          }
        }
      }
    }
  }
}

void RuntimeGraphShape::InitOperatorOutputTensor(
    const std::vector<pnnx::Operator *> &pnnx_operators,
    const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {
  CHECK(!pnnx_operators.empty() && !operators.empty());
  CHECK(pnnx_operators.size() == operators.size());
  for (uint32_t i = 0; i < pnnx_operators.size(); ++i) {
    const std::vector<pnnx::Operand *> operands = pnnx_operators.at(i)->outputs;
    CHECK(operands.size() <= 1) << "Only support one node one output yet!";
    if (operands.empty()) {
      continue;
    }
    CHECK(operands.size() == 1) << "Only support one output in the KuiperInfer";
    pnnx::Operand *operand = operands.front();
    const auto &runtime_op = operators.at(i);
    CHECK(operand != nullptr) << "Operand output is null";
    const std::vector<int32_t> &operand_shapes = operand->shape;
    const auto &output_tensors = runtime_op->output_operands;

    const int32_t batch = operand_shapes.at(0);
    CHECK(batch >= 0) << "Dynamic batch size is not supported!";
    CHECK(operand_shapes.size() == 2 || operand_shapes.size() == 4 ||
        operand_shapes.size() == 3)
            << "Unsupported shape sizes: " << operand_shapes.size();

    if (!output_tensors) {
      std::shared_ptr<RuntimeOperand> output_operand =
          std::make_shared<RuntimeOperand>();
      output_operand->shapes = operand_shapes;
      output_operand->type = RuntimeDataType::kTypeFloat32;
      output_operand->name = operand->name + "_output";
      for (int j = 0; j < batch; ++j) {
        if (operand_shapes.size() == 4) {
          output_operand->datas.push_back(std::make_shared<Tensor<float>>(
              operand_shapes.at(1), operand_shapes.at(2),
              operand_shapes.at(3)));
        } else if (operand_shapes.size() == 2) {
          output_operand->datas.push_back(
              std::make_shared<Tensor<float>>(1, operand_shapes.at(1), 1));
        } else {
          // current shape is 3
          output_operand->datas.push_back(std::make_shared<Tensor<float>>(
              1, operand_shapes.at(1), operand_shapes.at(2)));
        }
      }
      runtime_op->output_operands = std::move(output_operand);
    } else {
      CHECK(batch == output_tensors->datas.size());
      // output_tensors empty
      CHECK(output_tensors->type == RuntimeDataType::kTypeFloat32);
      CHECK(output_tensors->shapes == operand_shapes);
      for (uint32_t b = 0; b < batch; ++b) {
        const std::vector<uint32_t> &tensor_shapes =
            output_tensors->datas.at(b)->shapes();
        if (operand_shapes.size() == 4) {
          if (tensor_shapes.at(0) != operand_shapes.at(1) ||
              tensor_shapes.at(1) != operand_shapes.at(2) ||
              tensor_shapes.at(2) != operand_shapes.at(3)) {
            DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
            const auto &target_shapes = std::vector<uint32_t>{(uint32_t) operand_shapes.at(1),
                                                              (uint32_t) operand_shapes.at(2),
                                                              (uint32_t) operand_shapes.at(3)};
            output_tensors->datas.at(b)->ReRawshape(target_shapes);
          }
        } else if (operand_shapes.size() == 2) {
          if (tensor_shapes.at(0) != 1 ||
              tensor_shapes.at(1) != operand_shapes.at(1) ||
              tensor_shapes.at(2) != 1) {
            DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
            const auto &target_shapes = std::vector<uint32_t>{1, (uint32_t) operand_shapes.at(1), 1};
            output_tensors->datas.at(b)->ReRawshape(target_shapes);
          }
        } else {
          // current shape is 3
          if (tensor_shapes.at(0) != 1 ||
              tensor_shapes.at(1) != operand_shapes.at(1) ||
              tensor_shapes.at(2) != operand_shapes.at(2)) {
            DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
            const auto &target_shapes =
                std::vector<uint32_t>{1, (uint32_t) operand_shapes.at(1), (uint32_t) operand_shapes.at(2)};
            output_tensors->datas.at(b)->ReRawshape(target_shapes);
          }
        }
      }
    }
  }
}

RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {

}

void RuntimeGraph::set_bin_path(const std::string &bin_path) {
  this->bin_path_ = bin_path;
}

void RuntimeGraph::set_param_path(const std::string &param_path) {
  this->param_path_ = param_path;
}

const std::string &RuntimeGraph::param_path() const {
  return this->param_path_;
}

const std::string &RuntimeGraph::bin_path() const {
  return this->bin_path_;
}

bool RuntimeGraph::Init() {
  if (this->bin_path_.empty() || this->param_path_.empty()) {
    LOG(ERROR) << "The bin path or param path is empty";
    return false;
  }

  this->graph_ = std::make_unique<pnnx::Graph>();
  int load_result = this->graph_->load(param_path_, bin_path_);
  if (load_result != 0) {
    LOG(ERROR) << "Load param path and bin path error: " << param_path_ << " " << bin_path_;
    return false;
  }

  //
  std::vector<pnnx::Operator *> operators = this->graph_->ops;
  if (operators.empty()) {
    LOG(ERROR) << "Can not read the layers' define";
    return false;
  }

  this->operators_.clear();
  // 根据const pnnx::Operator *op 去赋值std::shared_ptr<RuntimeOperator> runtime_operator
  for (const pnnx::Operator *op : operators) {
    if (!op) {
      LOG(ERROR) << "Meet the empty node";
      continue;
    } else {
      // 现在是空的，下面
      std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();
      // 初始化算子的名称
      runtime_operator->name = op->name;
      runtime_operator->type = op->type;

      // 初始化算子中的input，对操作符号operator赋予runtimeoperand作为输入，输入是根据pnnx::operand来的
      const std::vector<pnnx::Operand *> &inputs = op->inputs;
      if (!inputs.empty()) {
        InitInputOperators(inputs, runtime_operator);
      }

      // 记录输出operand中的名称
      // 有一个pnnx::operator 来自与load_graph这个操作
      // load_graph pnnx::operators数组 进行遍历 pnnx::operator
      // 每一个遍历中operator，我们再初始化自己的kuiperinfer::RuntimeOperator

      /// RuntimeOperator根据pnnx::operator赋予inputs和outputs
      const std::vector<pnnx::Operand *> &outputs = op->outputs;
      if (!outputs.empty()) {
        InitOutputOperators(outputs, runtime_operator);
      }

      // 初始化算子中的attribute(权重)
      //没一个pnnx::operator里面有一个权重，我们根据pnnx::Attr这个权重去初始化RuntimeAttr
      /// 初始化RutimeAttr之后呢，存放在runtime_operator
      const std::map<std::string, pnnx::Attribute> &attrs = op->attrs;
      if (!attrs.empty()) {
        InitGraphAttrs(attrs, runtime_operator);
      }

      // 初始化算子中的parameter
      // 根据const pnnx::Operator *op 去赋值std::shared_ptr<RuntimeOperator> runtime_operator
      // 先得到pnnx::parameter再根据这个去赋值RuntimeOperator中的RuntimeParameter
      const std::map<std::string, pnnx::Parameter> &params = op->params;
      // op是pnnx来的，我们需要将pnnx的prams存放到runtime_oprator中
      if (!params.empty()) {
        InitGraphParams(params, runtime_operator);
      }
      // runtime_operator初始化玩成了
      this->operators_.push_back(runtime_operator);
    }
  }

  // 构建图关系
  for (const auto &current_op : this->operators_) {
    const std::vector<std::string> &output_names = current_op->output_names;
    for (const auto &next_op : this->operators_) {
      if (next_op == current_op) {
        continue;
      }
      if (std::find(output_names.begin(), output_names.end(), next_op->name) !=
          output_names.end()) {
        current_op->output_operators.insert({next_op->name, next_op});
      }
    }
  }
  graph_state_ = GraphState::NeedBuild;
  return true;
}

void RuntimeGraph::InitInputOperators(const std::vector<pnnx::Operand *> &inputs,
                                      const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const pnnx::Operand *input : inputs) {
    if (!input) {
      continue;
    }
    const pnnx::Operator *producer = input->producer;
    std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
    runtime_operand->name = producer->name;
    runtime_operand->shapes = input->shape;

    switch (input->type) {
      case 1: {
        runtime_operand->type = RuntimeDataType::kTypeFloat32;
        break;
      }
      case 0: {
        runtime_operand->type = RuntimeDataType::kTypeUnknown;
        break;
      }
      default: {
        LOG(FATAL) << "Unknown input operand type: " << input->type;
      }
    }
    runtime_operator->input_operands.insert({producer->name, runtime_operand});
    runtime_operator->input_operands_seq.push_back(runtime_operand);
  }
}

void RuntimeGraph::InitOutputOperators(const std::vector<pnnx::Operand *> &outputs,
                                       const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const pnnx::Operand *output : outputs) {
    if (!output) {
      continue;
    }
    const auto &consumers = output->consumers;
    for (const auto &c : consumers) {
      runtime_operator->output_names.push_back(c->name);
    }
  }
}

// pnnx的params转换为runtimeoperator runtimeparams 并放在runtimeoprator中
void RuntimeGraph::InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                                   const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const auto &pair : params) {
    const std::string &name = pair.first;
    const pnnx::Parameter &parameter = pair.second;
    const int type = parameter.type;
    // 根据传入的pnnx:params 进行遍历，每次遍历得到一些属性，并根据属性去初始化具体的RuntimeParameter
    // RuntimeParameter再存放到RutimeOperator中
    switch (type) {
      case int(RuntimeParameterType::kParameterUnknown): {
        RuntimeParameter *runtime_parameter = new RuntimeParameter;
        // 存入参数的名字和具体的类实例
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterBool): {
        RuntimeParameterBool *runtime_parameter = new RuntimeParameterBool;
        runtime_parameter->value = parameter.b;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterInt): {
        RuntimeParameterInt *runtime_parameter = new RuntimeParameterInt;
        runtime_parameter->value = parameter.i;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterFloat): {
        RuntimeParameterFloat *runtime_parameter = new RuntimeParameterFloat;
        runtime_parameter->value = parameter.f;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterString): {
        RuntimeParameterString *runtime_parameter = new RuntimeParameterString;
        runtime_parameter->value = parameter.s;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterIntArray): {
        RuntimeParameterIntArray *runtime_parameter = new RuntimeParameterIntArray;
        runtime_parameter->value = parameter.ai;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterFloatArray): {
        RuntimeParameterFloatArray *runtime_parameter = new RuntimeParameterFloatArray;
        runtime_parameter->value = parameter.af;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      case int(RuntimeParameterType::kParameterStringArray): {
        RuntimeParameterStringArray *runtime_parameter = new RuntimeParameterStringArray;
        runtime_parameter->value = parameter.as;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      default: {
        LOG(FATAL) << "Unknown parameter type";
      }
    }
  }
}

void RuntimeGraph::InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                                  const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const auto &pair : attrs) {
    const std::string &name = pair.first;
    const pnnx::Attribute &attr = pair.second;
    switch (attr.type) {
      case 1: {
        std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>();
        runtime_attribute->type = RuntimeDataType::kTypeFloat32;
        runtime_attribute->weight_data = attr.data;
        runtime_attribute->shape = attr.shape;
        runtime_operator->attribute.insert({name, runtime_attribute});
        break;
      }
      default : {
        LOG(FATAL) << "Unknown attribute type";
      }
    }
  }
}

const std::vector<std::shared_ptr<RuntimeOperator>> RuntimeGraph::operators() const {
  CHECK(graph_state_ == GraphState::Complete);
  return this->operators_;
}

void RuntimeGraph::Build(const std::string &input_name, const std::string &output_name) {
  if (graph_state_ == GraphState::NeedInit) {
    bool init_graph = Init();
    LOG_IF(FATAL, !init_graph) << "Init graph failed!";
  }

  CHECK(graph_state_ >= GraphState::NeedBuild)
          << "Graph status error, current state is " << int(graph_state_);
  LOG_IF(FATAL, this->operators_.empty())
          << "Graph operators is empty, may be no init";

  this->input_operators_maps_.clear();
  this->output_operators_maps_.clear();

  for (const auto &kOperator : this->operators_) {
    if (kOperator->type == "pnnx.Input") {
      this->input_operators_maps_.insert({kOperator->name, kOperator});
    } else if (kOperator->type == "pnnx.Output") {
      this->output_operators_maps_.insert({kOperator->name, kOperator});
    } else {
      // 以后的课中加layer的
    }
  }
  RuntimeGraphShape::InitOperatorInputTensor(operators_);
  RuntimeGraphShape::InitOperatorOutputTensor(graph_->ops, operators_);
  graph_state_ = GraphState::Complete;
  input_name_ = input_name;
  output_name_ = output_name;
}
}