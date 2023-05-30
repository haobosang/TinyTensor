find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(benchmark REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

aux_source_directory(../test/ DIR_TEST)

set(link_lib glog::glog GTest::gtest)
set(link_bench benchmark::benchmark)
set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_executable(test_TinyTensor ./test_main.cpp ${DIR_TEST})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native")

target_link_libraries(test_TinyTensor ${OpenCV_LIBS} TinyTensor)
target_link_libraries(test_TinyTensor ${link_lib} ${link_math_lib} ${link_bench})
target_link_directories(test_TinyTensor PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_TinyTensor TinyTensor)

target_include_directories(test_TinyTensor PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(test_TinyTensor PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_TinyTensor PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(test_TinyTensor PUBLIC ${Armadillo_INCLUDE_DIR})

set(CMAKE_BUILD_TYPE Debug)