
#include "runtime/runtime_ir.hpp"
#include <memory>
#include <iostream>
#include <iomanip>
#include <queue>
#include <deque>
#include <utility>
#include "factory/layer_factory.hpp"

namespace TinyTensor {
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
      if (!params.empty()) {
        InitGraphParams(params, runtime_operator);
      }
      // runtime_operator初始化玩成了
      this->operators_.push_back(runtime_operator);
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
  return this->operators_;
}
}