

#include "expression.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <stack>

namespace TinyTensor {
ExpressionLayer::ExpressionLayer(const std::string &statement)
    : Layer("Expression"), parser_(std::make_unique<ExpressionParser>(statement)) {

}

InferStatus ExpressionLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                     std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of expression layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "Output and input size of the expression layer is not adapting";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  CHECK(this->parser_ != nullptr) << "The parser in the expression layer is null!";
  this->parser_->Tokenizer(false);
  const auto &expressions = this->parser_->tokens();
  CHECK(!expressions.empty()) << "The expression in the expression layer tokenize failed!";

  for (uint32_t i = 0; i < inputs.size(); ++i) {
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR) << "The input feature map of expression layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }
  }

  const uint32_t batch_size = outputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    if (outputs.at(i) == nullptr || outputs.at(i)->empty()) {
      LOG(ERROR) << "The output of the expression layer is empty";
      return InferStatus::kInferFailedInputOutSizeAdaptingError;
    }
    outputs.at(i)->Fill(0.f);
  }

  std::stack<std::vector<std::shared_ptr<Tensor<float>>>> op_stack;
  const std::vector<std::shared_ptr<TokenNode>> &token_nodes = this->parser_->Generate();
  for (const auto &token_node : token_nodes) {
    if (token_node->num_index >= 0) {
      // process operator
      uint32_t start_pos = token_node->num_index * batch_size;
      std::vector<std::shared_ptr<Tensor<float>>> input_token_nodes;
      for (uint32_t i = 0; i < batch_size; ++i) {
        CHECK(i + start_pos < inputs.size());
        input_token_nodes.push_back(inputs.at(i + start_pos));
      }
      op_stack.push(input_token_nodes);
    } else {
      // process operation
      const int32_t op = token_node->num_index;
      CHECK(op_stack.size() >= 2) << "The number of operand is less than two";
      std::vector<std::shared_ptr<Tensor<float>>> input_node1 = op_stack.top();

      CHECK(input_node1.size() == batch_size);
      op_stack.pop();

      std::vector<std::shared_ptr<Tensor<float>>> input_node2 = op_stack.top();
      CHECK(input_node2.size() == batch_size);
      op_stack.pop();

      std::vector<std::shared_ptr<Tensor<float>>> output_token_nodes(batch_size);
#pragma omp parallel for num_threads(batch_size)
      for (uint32_t i = 0; i < batch_size; ++i) {
        // do execution
        if (op == int(TokenType::TokenAdd)) {
          output_token_nodes.at(i) = TensorElementAdd(input_node1.at(i), input_node2.at(i));
        } else if (op == int(TokenType::TokenMul)) {
          output_token_nodes.at(i) = TensorElementMultiply(input_node1.at(i), input_node2.at(i));
        } else {
          LOG(FATAL) << "Unknown operator type: " << op;
        }
      }
      op_stack.push(output_token_nodes);
    }
  }

  CHECK(op_stack.size() == 1);
  std::vector<std::shared_ptr<Tensor<float>>> output_node = op_stack.top();
  op_stack.pop();
  for (int i = 0; i < batch_size; ++i) {
    CHECK(outputs.at(i) != nullptr && !outputs.at(i)->empty());
    CHECK(outputs.at(i)->shapes() == output_node.at(i)->shapes());
    outputs.at(i) = output_node.at(i);
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus ExpressionLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                      std::shared_ptr<Layer> &expression_layer) {
  CHECK(op != nullptr) << "Expression operator is nullptr";
  const auto &params = op->params;
  if (params.find("expr") == params.end()) {
    return ParseParameterAttrStatus::kParameterMissingExpr;
  }

  const auto &statement_param = dynamic_cast<RuntimeParameterString *>(params.at("expr"));
  if (statement_param == nullptr) {
    LOG(ERROR) << "Can not find the expression parameter";
    return ParseParameterAttrStatus::kParameterMissingExpr;
  }
  if (statement_param->type != RuntimeParameterType::kParameterString) {
    LOG(ERROR) << "Can not find the expression parameter";
    return ParseParameterAttrStatus::kParameterMissingExpr;
  }

  expression_layer = std::make_shared<ExpressionLayer>(statement_param->value);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kExpressionGetInstance("pnnx.Expression", ExpressionLayer::GetInstance);
}
