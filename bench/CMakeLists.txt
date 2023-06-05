find_package(benchmark REQUIRED)
aux_source_directory(../bench DIR_BENCH)

set(link_lib benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_TinyTensor ${DIR_BENCH})
target_link_directories(bench_TinyTensor PUBLIC ${PROJECT_SOURCE_DIR}/lib)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
endif()

target_link_directories(bench_TinyTensor PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(bench_TinyTensor ${link_lib} OpenMP::OpenMP_CXX)
target_link_libraries(bench_TinyTensor TinyTensor)

# if (MSVC)
#     # find kuiper dll
#     add_custom_command(TARGET bench_kuiper POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#             "$<TARGET_FILE_DIR:kuiper>/kuiper.dll"
#             $<TARGET_FILE_DIR:bench_kuiper>)
# endif()

target_include_directories(bench_TinyTensor PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(bench_TinyTensor PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(bench_TinyTensor PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(bench_TinyTensor PUBLIC ${Armadillo_INCLUDE_DIR})

# if(CMAKE_BUILD_TYPE MATCHES Release)
#   set_target_properties(bench_TinyTensor PROPERTIES COMPILE_FLAGS "-Wno-unused-command-line-argument")
#   set_target_properties(bench_TinyTensor PROPERTIES LINK_FLAGS "-Wl,--allow-multiple-definition -Wl,--whole-archive -lbenchmark -Wl,--no-whole-archive -Wl,--export-dynamic -Wl,--dynamic-list=${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols.lds")
# endif()