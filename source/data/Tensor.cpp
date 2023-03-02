#include "data/Tensor.hpp"
#include<glog/logging.h>
#include<memory>

namespace TinyTensor{

Tensor<float>::Tensor(uint32_t channels,uint32_t rows,uint32_t cols){
    data_ = arma::fcube(channels,rows,cols);
}
Tensor<float>::Tensor(const Tensor &tensor){
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;

}
Tensor<float> &Tensor<float>::operator=(const Tensor &tensor){
    return *this;

}
void Tensor<float>::set_size(const arma::fcube &data){
    this->data_ = data;

}

bool Tensor<float>::empty() const{
    return this->data_.empty();

}
uint32_t Tensor<float>::channels() const{
    return this->data_.n_slices;

}
uint32_t Tensor<float>::rows() const{
    return this->data_.n_rows;

}
uint32_t Tensor<float>::cols() const{
    return this->data_.n_cols;

}
uint32_t Tensor<float>::size() const{
    return this->data_.size();

}


}
