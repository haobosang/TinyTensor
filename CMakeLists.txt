cmake_minimum_required(VERSION 3.16)
project(MyTinyTensor)
set(CMAKE_CXX_STANDARD 17)
include_directories(./include)
option(BUILD_DEMO "BUILD THE DEMO PROJECT")
set(BUILD_DEMO ON)

find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

aux_source_directory(./source/data DIR_DATA)
aux_source_directory(./source/runtime DIR_PARSER)
aux_source_directory(./source/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(./source/layer/details DIR_BINOCULAR_LAYER)
aux_source_directory(./source/parser DIR_PARSER)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(link_lib glog::glog)

set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_library(TinyTensor SHARED ${DIR_DATA} ${DIR_PARSER} ${DIR_ABSTRACT_LAYER} ${DIR_BINOCULAR_LAYER} ${DIR_PARSER} )
target_link_libraries(TinyTensor ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(TinyTensor PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(TinyTensor PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(TinyTensor PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(TinyTensor PUBLIC ${Armadillo_INCLUDE_DIR})

# mathfun library defines
add_compile_definitions(SSE_MATHFUN_WITH_CODE USE_SSE_AUTO)
enable_testing()
add_subdirectory(test)
add_subdirectory(bench)

set(CMAKE_BUILD_TYPE Debug)