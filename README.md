<!--
 * @Author: lihaobo
 * @Date: 2023-03-02 10:12:49
 * @LastEditors: lihaobo
 * @LastEditTime: 2023-05-10 08:17:41
 * @Description: 请填写简介
-->


<p align="center">
  <img width="100%" src="./img/TinyTensor.png" alt="Banner">
</p>
<p align="center">
  <b>
TinyTensor is an efficient lightweight deep learning inference framework.</b>
</p>
<p align="center">
  <a href="https://github.com/haobosang/TinyTensor/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/koekeishiya/yabai.svg?color=green" alt="License Badge">
  </a>
  <a href="https://github.com/haobosang/TinyTensor/tree/main/docs">
    <img src="https://img.shields.io/badge/view-documentation-green.svg" alt="Documentation Badge">
  </a>
  <a href="https://github.com/haobosang/TinyTensor">
    <img src="https://img.shields.io/badge/cmake-3.16%2B-green" alt="cmake Badge">
  </a>
  
  
</p>

## About
TinyTensor supports a variety of popular neural network architectures such as convolutional neural networks (CNNs), recurrent neural networks (RNNs), and fully connected neural networks, and it can be used for tasks such as image classification, object detection, speech recognition, and natural language processing.
## Development Environment
* Development   language: C++ 20
* Math Library: Armadillo
* Logging framework：Google glog
* Unit test:    Google Test
* Code style:   Clang format
* Performance testing： Benckmark

## How to build on Linux
### Ubuntu 18 (Debian 10)
```
apt update
apt install cmake libopenblas-dev liblapack-dev \
libarpack2-dev libsuperlu-dev libomp-dev libopencv-dev
```
### Install Armadillo
```
wget https://sourceforge.net/projects/arma/files/armadillo-12.2.0.tar.xz
mkdir build && cd build
cmake ..
make -j8
make install
```
### Install Benchmark
```
cd third_party
git submodule update --init
mv googletest benchmark
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE ../benchmark
make -j8
# 如果想全局安装就接着运行下面的命令
sudo make install
```
## Operators Currently Implemented
* ReLU
* Sigmoid
* Conv
* MaxPooling

## Performance Testing
### Test Equipment

Intel(R) Xeon(R) W-2223 CPU @ 3.60GHz


### Compilation Environment

gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0

### Performance Results

| **input size**         | **model**        | **Computing Device**              | **time**           |
|------------------------| ---------------- | ------------------------- |------------------|
| 224×224 batch = 8      | ResNet18         | CPU(armadillo + openblas) | 59.75ms / image  |
| 224×224 batch =16      | ResNet18         | CPU(armadillo + openblas) | 28.12ms / image  |

## Acknowledgement
caffe