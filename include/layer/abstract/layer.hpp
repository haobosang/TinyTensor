/*
 * @Author: haobosang lihaobo1998@gmail.com
 * @Date: 2023-03-06 10:04:15
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-06 17:43:25
 * @FilePath: /MyTinyTensor/include/layer/layer.hpp
 * @Description:
 */
#ifndef TINYTENSOR_INCLUDE_LAYER_HPP_
#define TINYTENSOR_INCLUDE_LAYER_HPP_

#include "data/Tensor.hpp"
#include <string>
#include "status_code.hpp"
#include <vector>

namespace TinyTensor {

class Layer {
private:
  /* data */
  std::string _layer_name_;

public:
  explicit Layer(const std::string &layer_name):_layer_name_(std::move(layer_name)){

  }

  virtual ~Layer() = default;

  virtual InferStatus Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
           std::vector<std::shared_ptr<Tensor<float>>> &outputs);

  /**
   * 返回层的权重
   * @return 返回的权重
   */
  virtual const std::vector<std::shared_ptr<Tensor<float>>> &weight() const;
   /**
   * 返回层的偏移量
   * @return 返回的偏移量
   */
  virtual const std::vector<std::shape_str<Tensor<float>>> &bias() const;

  virtual void set_weights(const std::vector<std::shared_ptr<Tensor<float>>> &weights);

  /**
   * 设置Layer的偏移量
   * @param bias 偏移量
   */
  virtual void set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &bias);

  /**
   * 设置Layer的权重
   * @param weights 权重
   */
  virtual void set_weights(const std::vector<float> &weights);

  /**
   * 设置Layer的偏移量
   * @param bias 偏移量
   */
  virtual void set_bias(const std::vector<float> &bias);

  /**
   * 返回层的名称
   * @return 层的名称
   */
  virtual const std::string &layer_name() const { return this->layer_name_; }



};

} // namespace TinyTensor

#endif // TINYTENSOR_INCLUDE_LAYER_HPP_