/*
 * @Author: lihaobo
 * @Date: 2023-05-10 01:39:47
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-05-11 05:31:08
 * @Description: 
 */

#ifndef TINYTENSOR_INCLUDE_TENSOR_HPP_
#define TINYTENSOR_INCLUDE_TENSOR_HPP_

#include <armadillo>
#include <memory>
#include <vector>

namespace TinyTensor {

template <typename T> class Tensor {};
template <> class Tensor<uint8_t> {
  // 暂时搁置
};
template <> class Tensor<float> {
public:
  explicit Tensor() = default;
  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
  explicit Tensor(const std::vector<uint32_t>& shapes);
  Tensor(const Tensor &tensor);
  Tensor<float> &operator=(const Tensor &tensor);

  uint32_t channels() const;
  uint32_t rows() const;
  uint32_t cols() const;
  uint32_t size() const;

  void set_data(const arma::fcube &data);
  bool empty() const;

  float index(uint32_t offset) const;
  float &index(uint32_t offset);

  std::vector<uint32_t> shapes() const;

  arma::fcube &data();

  const arma::fcube &data() const;

  arma::fmat &at(uint32_t channel);

  const arma::fmat &at(uint32_t channel) const;

  float at(uint32_t channel, uint32_t row, uint32_t col) const;

  float &at(uint32_t channel, uint32_t row, uint32_t col);

  void Padding(const std::vector<uint32_t> &pads, float padding_value);

  void Fill(float value);

  void Fill(const std::vector<float> &values);

  void Ones();

  void Rand();

  void Show();

  void Flatten();
  
  /**
   * @description: 
   * @param {uint32_t} channel
   * @return 返回fmat矩阵
   */
  arma::fmat& slice(uint32_t channel);

  /**
   * 返回张量第channel通道中的数据
   * @param channel 需要返回的通道
   * @return 返回的通道
   */
  const arma::fmat& slice(uint32_t channel) const;

  void ReRawshape(const std::vector<uint32_t> &shapes);

  void ReRawView(const std::vector<uint32_t> &shapes);

  std::shared_ptr<Tensor<float>> Clone();


private:
  void ReView(const std::vector<uint32_t> &shapes);
  arma::fcube data_;
  std::vector<uint32_t> raw_shapes_;
};

std::tuple<std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>> TensorBroadcast(const std::shared_ptr<Tensor<float>> &s1, const std::shared_ptr<Tensor<float>> &s2);

std::shared_ptr<Tensor<float>> TensorPadding(
    const std::shared_ptr<Tensor<float>>& tensor,
    const std::vector<uint32_t>& pads, float padding_value);

/**
 * 比较tensor的值是否相同
 * @param a 输入张量1
 * @param b 输入张量2
 * @return 比较结果
 */
bool TensorIsSame(const std::shared_ptr<Tensor<float>>& a,
                  const std::shared_ptr<Tensor<float>>& b);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @return 张量相加的结果
 */
std::shared_ptr<Tensor<float>> TensorElementAdd(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @param output_tensor 输出张量
 */
void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                      const std::shared_ptr<Tensor<float>>& tensor2,
                      const std::shared_ptr<Tensor<float>>& output_tensor);

/**
 * 矩阵点乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @param output_tensor 输出张量
 */
void TensorElementMultiply(const std::shared_ptr<Tensor<float>>& tensor1,
                           const std::shared_ptr<Tensor<float>>& tensor2,
                           const std::shared_ptr<Tensor<float>>& output_tensor);

/**
 * 张量相乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @return 张量相乘的结果
 */
std::shared_ptr<Tensor<float>> TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2);

/**
 * 创建一个张量
 * @param channels 通道数量
 * @param rows 行数
 * @param cols 列数
 * @return 创建后的张量
 */
std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                            uint32_t cols);

/**
 * 创建一个张量
 * @param shapes 张量的形状
 * @return 创建后的张量
 */
std::shared_ptr<Tensor<float>> TensorCreate(
    const std::vector<uint32_t>& shapes);


} // namespace TinyTensor

#endif // TINTTENSOR_INCLUDE_TENSOR_HPP_