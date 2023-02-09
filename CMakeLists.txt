cmake_minimum_required(VERSION 3.24)
project(thread_pool_library)

set(CMAKE_CXX_STANDARD 17)

add_executable(thread_pool thread_pool_test.cpp)
add_executable(thread_puddle thread_puddle_test.cpp)
