find_package(GTest REQUIRED)

aux_source_directory(../test DIR_TEST)
set(link_lib glog gtest pthread  )
set(link_math_lib armadillo blas lapack)

aux_source_directory(../source/data DIR_DATA)
add_executable(test_TinyTensor ${DIR_TEST} ${DIR_DATA})
#add_executable(test_TinyTensor test_tensor.cpp)
link_directories(/usr/local/lib/)   
target_link_libraries(test_TinyTensor ${link_lib} ${link_math_lib})