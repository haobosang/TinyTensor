#include "data/load_data.hpp"
#include <memory>
#include <gtest/gtest.h>
#include <armadillo>
#include <glog/logging.h>
#include <vector>
using namespace TinyTensor;
int main(){
    const std::string &file_path = "../tmp/data1.csv";
    std::shared_ptr<Tensor<float>> data = CSVDataLoader::LoadData(file_path, ',');
    uint32_t index = 1;
    uint32_t rows = data->rows();
    uint32_t cols = data->cols();
    std::cout<<rows<<std::endl;
    std::cout<<cols<<std::endl;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
        std::cout<<data->at(0, r, c);
        index += 1;
        }
    return 0;
    }
}
