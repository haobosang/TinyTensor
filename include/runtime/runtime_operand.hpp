/*
 * @Author: lihaobo
 * @Date: 2023-03-11 13:46:55
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-11 14:47:37
 * @Description: 请填写简介
 */

#ifndef TINYTENSOR_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#define TINYTENSOR_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#include "data/Tensor.hpp"
#include "runtime_datatype.hpp"
#include "status_code.hpp"
#include <memory>
#include <string>
#include <vector>

namespace TinyTensor {
/// 计算节点输入输出的操作数
struct RuntimeOperand {
  std::string name;                                  /// 操作数的名称
  std::vector<int32_t> shapes;                       /// 操作数的形状
  std::vector<std::shared_ptr<Tensor<float>>> datas; /// 存储操作数
  RuntimeDataType type =
      RuntimeDataType::kTypeUnknown; /// 操作数的类型，一般是float
};
} // namespace TinyTensor
#endif // TINYTENSOR_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
