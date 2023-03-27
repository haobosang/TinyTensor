/*
 * @Author: lihaobo
 * @Date: 2023-03-21 19:52:25
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-03-27 16:57:06
 * @Description: 请填写简介
 */
#include <gtest/gtest.h>
#include <glog/logging.h>

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("MyTinyTensor");
    FLAGS_log_dir = "./log/";
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Start test...\n";
    return RUN_ALL_TESTS();
}