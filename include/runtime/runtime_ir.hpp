#include <glog/logging.h>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "factory/layer_factory.hpp"
#include "ir.h"
#include "runtime/runtime_operand.hpp"
#include "runtime_op.hpp"

// runtimegraph -->> pnnx::graph
// runtimeOperator -->> pnnx::operator
// runtimeOperand -->> pnnx::operand
// runtimeParameter --> pnnx::Params
// runtimeAttrs -->> pnnx::attrs、
//   int load_result = this->graph_->load(param_path_, bin_path_);
//   得到pnnx::graph,一个一个算子去初始化 runtimegraph
//  得到pnnx::operators -->依次便利
// pnnx operator 遍历，初始化一个RuntimeOperator
// 1.  得到pnnx
// operator的inputs，再根据这个inputs去初始化我们RuntimeOperator::runtime_operator->input_operands
// 2. 同理得到pnnx
// operator的outputs去初始化RuntimeOperator::runtime_operator->output_operands
// 3. runtimeParameter 根据pnnx::param初始化
// 4, runtimeAttr根据pnnx::attr初始化
// 1.2.3.4的初始化过程中，
// runtimeParameter，runtimeAttr，output_operands.inputs_operand放在一个runtime_operator里面

// 5. 再把这个runtime_operator存放好
// 这个runtime_operator中既有输入的参数，又有输出的数、参数，又有层的参数，又有层的权重！
// 转换成功！

namespace TinyTensor {
/// 计算图结构，由多个计算节点和节点之间的数据流图组成

class RuntimeGraphShape {
public:
  /**
   * 如果图是第一次运行，则根据节点输入operand的形状准备好后续Layer计算中所需要的Tensor
   * 如果图是第二次以上运行，则检查输入operand的形状和operand中张量的形状是否匹配
   * @param operators 计算图中的计算节点
   */
  static void InitOperatorInputTensor(
      const std::vector<std::shared_ptr<RuntimeOperator>> &operators);

  /**
   * 如果图是第一次运行，则根据节点输出operand的形状准备好后续Layer计算中所需要的Tensor
   * 如果图是第二次以上运行，则检查输出operand的形状和operand中张量的形状是否匹配
   * @param pnnx_operators pnnx图节点
   * @param operators KuiperInfer计算图中的计算节点
   */
  static void InitOperatorOutputTensor(
      const std::vector<pnnx::Operator *> &pnnx_operators,
      const std::vector<std::shared_ptr<RuntimeOperator>> &operators);
};

class RuntimeGraph {
public:
  /**
   * 计算图的初始化
   * @return 是否初始化成功
   */
  bool Init();

  void Build(const std::string &input_name, const std::string &output_name);
  /**
   * 初始化计算图
   * @param param_path 计算图的结构文件
   * @param bin_path 计算图中的权重文件
   */
  RuntimeGraph(std::string param_path, std::string bin_path);

  /**
   * 设置权重文件
   * @param bin_path 权重文件路径
   */
  void set_bin_path(const std::string &bin_path);

  /**
   * 设置结构文件
   * @param param_path  结构文件路径
   */
  void set_param_path(const std::string &param_path);

  /**
   * 返回结构文件
   * @return 返回结构文件
   */
  const std::string &param_path() const;

  /**
   * 返回权重文件
   * @return 返回权重文件
   */
  const std::string &bin_path() const;

  const std::vector<std::shared_ptr<RuntimeOperator>> operators() const;

private:
  /**
   * 初始化kuiper infer计算图节点中的输入操作数
   * @param inputs pnnx中的输入操作数
   * @param runtime_operator 计算图节点
   */
  static void
  InitInputOperators(const std::vector<pnnx::Operand *> &inputs,
                     const std::shared_ptr<RuntimeOperator> &runtime_operator);

  /**
   * 初始化kuiper infer计算图节点中的输出操作数
   * @param outputs pnnx中的输出操作数
   * @param runtime_operator 计算图节点
   */
  static void
  InitOutputOperators(const std::vector<pnnx::Operand *> &outputs,
                      const std::shared_ptr<RuntimeOperator> &runtime_operator);

  /**
   * 初始化kuiper infer计算图中的节点属性
   * @param attrs pnnx中的节点属性
   * @param runtime_operator 计算图节点
   */
  static void
  InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                 const std::shared_ptr<RuntimeOperator> &runtime_operator);

  /**
   * 初始化kuiper infer计算图中的节点参数
   * @param params pnnx中的参数属性
   * @param runtime_operator 计算图节点
   */
  static void
  InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                  const std::shared_ptr<RuntimeOperator> &runtime_operator);

private:
  enum class GraphState {
    NeedInit = -2,
    NeedBuild = -1,
    Complete = 0,
  };
  GraphState graph_state_ = GraphState::NeedInit;
  std::string input_name_;  /// 计算图输入节点的名称
  std::string output_name_; /// 计算图输出节点的名称
  std::string param_path_;  /// 计算图的结构文件
  std::string bin_path_;    /// 计算图的权重文件
  std::map<std::string, std::shared_ptr<RuntimeOperator>>
      input_operators_maps_; /// 保存输入节点
  std::map<std::string, std::shared_ptr<RuntimeOperator>>
      output_operators_maps_; /// 保存输出节点
  std::vector<std::shared_ptr<RuntimeOperator>> operators_; /// 计算图的计算节点
  std::unique_ptr<pnnx::Graph> graph_;                      /// pnnx的graph
};

} // namespace TinyTensor