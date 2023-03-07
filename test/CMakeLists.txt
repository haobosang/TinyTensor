find_package(GTest REQUIRED)

aux_source_directory(../test DIR_TEST)
set(link_lib glog gtest pthread  )
set(link_math_lib armadillo blas lapack)

#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

aux_source_directory(../source/data DIR_DATA)
aux_source_directory(../source/ops DIR_OP)
aux_source_directory(../source/layer DIR_LAYER)
aux_source_directory(../source/factory DIR_FAC)

#add_executable(test_TinyTensor ${DIR_TEST} ${DIR_DATA})
add_executable(test_TinyTensor ${DIR_TEST} ${DIR_DATA} ${DIR_OP} ${DIR_LAYER} ${DIR_FAC})
#add_executable(test_TinyTensor test_tensor.cpp)
link_directories(/usr/local/lib/)   
target_link_libraries(test_TinyTensor ${link_lib} ${link_math_lib})