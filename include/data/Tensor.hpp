/*
 * @Author: lihaobo
 * @Date: 2023-02-28 14:03:59
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-22 19:30:23
 * @Description: 请填写简介
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

  void ReRawshape(const std::vector<uint32_t> &shapes);

  void ReRawView(const std::vector<uint32_t> &shapes);

  std::shared_ptr<Tensor<float>> Clone();

  static std::shared_ptr<Tensor<float>>
  ElementAdd(const std::shared_ptr<Tensor<float>> &tensor1,
             const std::shared_ptr<Tensor<float>> &tensor2);

  static std::shared_ptr<Tensor<float>>
  ElementMul(const std::shared_ptr<Tensor<float>> &tensor1,
             const std::shared_ptr<Tensor<float>> &tensor2);

private:
  void ReView(const std::vector<uint32_t> &shapes);
  arma::fcube data_;
  std::vector<uint32_t> raw_shapes_;
};

} // namespace TinyTensor

#endif // TINTTENSOR_INCLUDE_TENSOR_HPP_