/*
 * @Author: lihaobo
 * @Date: 2023-03-11 13:46:51
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-11 14:33:56
 * @Description: 请填写简介
 */
#ifndef TINYTENSOR_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
#define TINYTENSOR_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_

#include "layer/abstract/layer.hpp"
#include "runtime/ir.h"
#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_parameter.hpp"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace TinyTensor {
class Layer;

/// 计算图中的计算节点
struct RuntimeOperator {
  int32_t meet_num = 0; /// 计算节点被相连接节点访问到的次数
  virtual ~RuntimeOperator();
  std::string name;             /// 计算节点的名称
  std::string type;             /// 计算节点的类型
  std::shared_ptr<Layer> layer; /// 节点对应的计算Layer

  std::vector<std::string> output_names; /// 节点的输出节点名称
  std::shared_ptr<RuntimeOperand> output_operands; /// 节点的输出操作数

  std::map<std::string, std::shared_ptr<RuntimeOperand>>
      input_operands; /// 节点的输入操作数
  std::vector<std::shared_ptr<RuntimeOperand>>
      input_operands_seq; /// 节点的输入操作数，顺序排列
  std::map<std::string, std::shared_ptr<RuntimeOperator>>
      output_operators; /// 输出节点的名字和节点对应

  std::map<std::string, RuntimeParameter *> params; /// 算子的参数信息
  std::map<std::string, std::shared_ptr<RuntimeAttribute>>
      attribute; /// 算子的属性信息，内含权重信息
};

class RuntimeOperatorUtils {
public:
  /**
   * 如果图是第一次运行，则根据节点输入operand的形状准备好后续Layer计算中所需要的Tensor
   * 如果图是第二次以上运行，则检查输入operand的形状和operand中张量的形状是否匹配
   * @param operators 计算图中的计算节点
   */
  static void InitOperatorInput(
      const std::vector<std::shared_ptr<RuntimeOperator>> &operators);

  /**
   * 如果图是第一次运行，则根据节点输出operand的形状准备好后续Layer计算中所需要的Tensor
   * 如果图是第二次以上运行，则检查输出operand的形状和operand中张量的形状是否匹配
   * @param pnnx_operators pnnx图节点
   * @param operators KuiperInfer计算图中的计算节点
   */
  static void InitOperatorOutput(
      const std::vector<pnnx::Operator *> &pnnx_operators,
      const std::vector<std::shared_ptr<RuntimeOperator>> &operators);
};

} // namespace TinyTensor
#endif // TINYTENSOR_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
