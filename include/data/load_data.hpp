#ifndef TINYTENSOR_INCLUDE_LOAD_DATA_HPP_
#define TINYTENSOR_INCLUDE_LOAD_DATA_HPP_

#include<armadillo>
#include "data/Tensor.hpp"

namespace TinyTensor
{

class CSVDataLoader
{
public:
    static arma::fmat LoadData(const std::string &file_path, char split_char = ',');
    
private:
    static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, char split_char);
};




} // namespace TinyTensor


#endif //TINYTENSOR_INCLUDE_LOAD_DATA_HPP_
