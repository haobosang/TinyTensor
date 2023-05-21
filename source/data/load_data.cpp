//
// Created by haobosang on 23-3-4.
//
#include "data/load_data.hpp"
#include <glog/logging.h>

namespace TinyTensor
{
    std::shared_ptr<Tensor<float>> CSVDataLoader::LoadDataWithHeader
    (
        const std::string &file_path,
        std::vector<std::string> &headers,
        char split_char
    )
    {
        CHECK(!file_path.empty()) << "File path is empty!";
        std::ifstream in(file_path);
        CHECK(in.is_open() && in.good()) << "File open failed! " << file_path;

        std::string line_str;
        std::stringstream line_stream;

        const auto &[rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
        CHECK(rows >= 1);
        std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float>>(1, rows - 1, cols);
        arma::fmat &data = input_tensor->at(0);

        size_t row = 0;
        while (in.good())
        {
            std::getline(in, line_str);
            if (line_str.empty())
            {
                break;
            }

            std::string token;
            line_stream.clear();
            line_stream.str(line_str);

            size_t col = 0;
            while (line_stream.good())
            {
                std::getline(line_stream, token, split_char);
                try
                {
                    // todo
                    //  能够读取到第一行的csv列名，并存放在headers中
                    if (row == 0)
                    {
                        headers.push_back(token);
                    }
                    else
                    {
                        data.at(row - 1, col) = std::stof(token);
                    }

                    // 能够读取到第二行之后的csv数据，并相应放置在data变量的row，col位置中
                }
                catch (std::exception &e)
                {
                    LOG(ERROR) << "Parse CSV File meet error: " << e.what();
                    continue;
                }
                col += 1;
                CHECK(col <= cols) << "There are excessive elements on the column";
            }

            row += 1;
            CHECK(row <= rows) << "There are excessive elements on the row";
        }
        return input_tensor;
    }

    std::shared_ptr<Tensor<float>> CSVDataLoader::LoadData
    (
        const std::string &file_path,
        char split_char
    )
    {
        CHECK(!file_path.empty()) << "File path is empty!";
        std::ifstream in(file_path);
        CHECK(in.is_open() && in.good()) << "File open failed! " << file_path;

        std::string line_str;
        std::stringstream line_stream;

        const auto &[rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
        std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float>>(1, rows, cols);
        arma::fmat &data = input_tensor->at(0);

        size_t row = 0;
        while (in.good())
        {
            std::getline(in, line_str);
            if (line_str.empty())
            {
                break;
            }

            std::string token;
            line_stream.clear();
            line_stream.str(line_str);

            size_t col = 0;
            while (line_stream.good())
            {
                std::getline(line_stream, token, split_char);
                try
                {
                    data.at(row, col) = std::stof(token);
                }
                catch (std::exception &e)
                {
                    LOG(ERROR) << "Parse CSV File meet error: " << e.what();
                    continue;
                }
                col += 1;
                CHECK(col <= cols) << "There are excessive elements on the column";
            }

            row += 1;
            CHECK(row <= rows) << "There are excessive elements on the row";
        }
        return input_tensor;
    }

    std::pair<size_t, size_t> CSVDataLoader::GetMatrixSize
    (
        std::ifstream &file, 
        char split_char
    )
    {
        bool load_ok = file.good();
        file.clear();
        size_t fn_rows = 0;
        size_t fn_cols = 0;
        const std::ifstream::pos_type start_pos = file.tellg();

        std::string token;
        std::string line_str;
        std::stringstream line_stream;

        while (file.good() && load_ok)
        {
            std::getline(file, line_str);
            if (line_str.empty())
            {
                break;
            }

            line_stream.clear();
            line_stream.str(line_str);
            size_t line_cols = 0;

            std::string row_token;
            while (line_stream.good())
            {
                std::getline(line_stream, row_token, split_char);
                ++line_cols;
            }
            if (line_cols > fn_cols)
            {
                fn_cols = line_cols;
            }

            ++fn_rows;
        }
        file.clear();
        file.seekg(start_pos);
        return {fn_rows, fn_cols};
    }
}
