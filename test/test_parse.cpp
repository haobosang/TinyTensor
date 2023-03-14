/*
 * @Author: lihaobo
 * @Date: 2023-03-14 09:59:13
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-14 10:13:52
 * @Description: 请填写简介
 */
#include "parser/parse_expression.hpp"
#include <gtest/gtest.h>
#include <glog/logging.h>

using namespace TinyTensor;

static void ShowNodes(const std::shared_ptr<TinyTensor::TokenNode> &node) {
  if (!node) {
    return;
  }
  // 中序遍历的顺序
  ShowNodes(node->left);
  if (node->num_idx < 0) {
    if (node->num_idx == -int(TinyTensor::TokenType::TokenAdd)) {
      LOG(INFO) << "ADD";
    } else if (node->num_idx == -int(TinyTensor::TokenType::TokenMul)) {
      LOG(INFO) << "MUL";
    }
  } else {
    LOG(INFO) << "NUM: " << node->num_idx;
  }
  ShowNodes(node->right);
}
int main(){
    const std::string &statement = "add(mul(@0,@1),mul(@2,add(@3,@4)))";
    ExpressionParser parser(statement);
    const auto &node_tokens = parser.Generate();
    ShowNodes(node_tokens);

}