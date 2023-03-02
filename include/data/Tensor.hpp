#ifndef TINYTENSOR_INCLUDE_TENSOR_HPP_
#define TINYTENSOR_INCLUDE_TENSOR_HPP_

#include<memory>
#include<vector>
#include<armadillo>


namespace TinyTensor{

template<typename T> 
class Tensor{
    

};
template<>
class Tensor<uint8_t>{
    //暂时搁置
};
template<>
class Tensor<float>{
public:
    explicit Tensor() = default;
    explicit Tensor(uint32_t channels,uint32_t rows,uint32_t cols);

    Tensor(const Tensor &tensor);
    Tensor<float> &operator= (const Tensor &tensor);

    uint32_t channels() const;
    uint32_t rows() const;
    uint32_t cols() const;
    uint32_t size() const;

    void set_size(const arma::fcube &data); 
    bool empty() const;



    

private:
    arma::fcube data_;
    std::vector<uint32_t> raw_shapes_;
};


}




#endif //TINTTENSOR_INCLUDE_TENSOR_HPP_