cmake_minimum_required(VERSION 3.16)
project(MyTinyTensor)

set(CMAKE_CXX_STANDARD 17)

find_package(glog REQUIRED)
include_directories(./include)

set(link_lib glog pthread gtest)
set(link_math_lib armadillo blas lapack)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#ADD_SUBDIRECTORY(src_dir bin)
add_executable(MyTinyTensor main.cpp)

target_include_directories(MyTinyTensor PUBLIC /usr/include/armadillo_bits)
target_include_directories(MyTinyTensor PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(MyTinyTensor ${link_lib} ${link_math_lib})

enable_testing()
add_subdirectory(test)