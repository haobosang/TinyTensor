#include "data/Tensor.hpp"
#include<glog/logging.h>
#include<memory>

namespace TinyTensor{

Tensor<float>::Tensor(uint32_t channels,uint32_t rows,uint32_t cols){
    data_ = arma::fcube(rows,cols,channels);
}
Tensor<float>::Tensor(const Tensor &tensor){
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;

}
Tensor<float> &Tensor<float>::operator=(const Tensor &tensor){
    if(this != &tensor){
        this->data_ = tensor.data_;
        this->raw_shapes_ = tensor.raw_shapes_;
    }
    return *this;



}
void Tensor<float>::set_data(const arma::fcube &data){
    CHECK(data.n_rows==this->data_.n_rows) << data.n_rows << " != " << this->data_.n_rows;
    CHECK(data.n_cols==this->data_.n_cols) << data.n_cols << " != " << this->data_.n_cols;
    CHECK(data.n_slices==this->data_.n_slices) << data.n_slices << " != " << this->data_.n_slices;
    this->data_ = data;
    return ;
}

bool Tensor<float>::empty() const{
    return this->data_.empty();

}
uint32_t Tensor<float>::channels() const{
    CHECK(!this->data_.empty());
    return this->data_.n_slices;

}
uint32_t Tensor<float>::rows() const{
    CHECK(!this->data_.empty());
    return this->data_.n_rows;

}
uint32_t Tensor<float>::cols() const{
    CHECK(!this->data_.empty());
    return this->data_.n_cols;

}
uint32_t Tensor<float>::size() const{
    CHECK(!this->data_.empty());
    return this->data_.size();

}
float Tensor<float>::index(uint32_t offset) const{

}
std::vector<uint32_t> Tensor<float>::shapes() const{

}
arma::fcube &Tensor<float>::data(){
    return this->data_;
}
const arma::fcube &Tensor<float>::data() const{

}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const{

}

float &Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col){

}

void Tensor<float>::Padding(const std::vector<uint32_t> &pads, float padding_value){

}

void Tensor<float>::Fill(float value){
    this->data_.fill();

}

void Tensor<float>::Fill(const std::vector<float> &values){

}

void Tensor<float>::Ones(){

}

void Tensor<float>::Rand(){

}

void Tensor<float>::Show(){

}

void Tensor<float>::Flatten(){

}

}
