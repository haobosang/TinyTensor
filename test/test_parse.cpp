/*
 * @Author: lihaobo
 * @Date: 2023-03-14 09:59:13
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-14 13:13:51
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
    } else if(node->num_idx == -int(TinyTensor::TokenType::TokenDiv)){
        LOG(INFO) << "DIV";
    }

    } else {
    LOG(INFO) << "NUM: " << node->num_idx;
    }
  ShowNodes(node->right);
}
static void printTree(const std::shared_ptr<TinyTensor::TokenNode> &node, int indent = 0) {
    if (node == nullptr) {
        return;
    }
    printTree(node->right, indent + 4);
    // if (node->num_idx < 0) {
    //     if (node->num_idx == -int(TinyTensor::TokenType::TokenAdd)) {
    //         std::cout << "ADD";
    //     } else if (node->num_idx == -int(TinyTensor::TokenType::TokenMul)) {
    //         std::cout << "MUL";
    //     } else if(node->num_idx == -int(TinyTensor::TokenType::TokenDiv)){
    //         std::cout << "DIV";
    //     }
    // }
    std::cout << std::string(indent, ' ') << node->num_idx << std::endl;
   
    printTree(node->left, indent + 4);
}
int main(){
    const std::string &statement = "add(div(@0,@1),@2)";
    //const std::string &statement ="add(mul(@0,@1),@2)";
    ExpressionParser parser(statement);
    const auto &node_tokens = parser.Generate();
    //ShowNodes(node_tokens);
    printTree(node_tokens,0);

}