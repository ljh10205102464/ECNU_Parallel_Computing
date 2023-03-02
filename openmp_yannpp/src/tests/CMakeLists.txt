cmake_minimum_required(VERSION 3.9)

project(yannpp_tests_project CXX)

message(STATUS "set compipler")
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
set(OpenMP_LIBRARIES "/opt/homebrew/opt/libomp/lib")
set(OpenMP_INCLUDES "/opt/homebrew/opt/libomp/include")

OPTION (USE_OpenMP "Use OpenMP to enamble <omp.h>" ON)

# Find OpenMP
if(APPLE AND USE_OpenMP)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
      set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
      set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
      set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
      set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
      set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    endif()
endif()

if(USE_OpenMP)
  find_package(OpenMP REQUIRED)
endif(USE_OpenMP)

if (OpenMP_FOUND)
    include_directories("${OpenMP_INCLUDES}")
    link_directories("${OpenMP_LIBRARIES}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OpenMP_FOUND)

set(SOURCES
    ${MNIST_SOURCE_DIR}/parsing/mnist_dataset.h
    ${MNIST_SOURCE_DIR}/parsing/mnist_dataset.cpp
    ${MNIST_SOURCE_DIR}/parsing/parsed_labels.h
    ${MNIST_SOURCE_DIR}/parsing/parsed_labels.cpp
    ${MNIST_SOURCE_DIR}/parsing/parsed_images.h
    ${MNIST_SOURCE_DIR}/parsing/parsed_images.cpp
    tests_main.cpp
    tests_mnist.cpp)

add_executable(yannpp_tests ${SOURCES})

# if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")
# endif()

FILE(TO_NATIVE_PATH ${DATA_DIR} DATA_ROOT)

if(WIN32)
  STRING(REGEX REPLACE "\\\\" "\\\\\\\\" DATA_ROOT ${DATA_ROOT} )
endif()

add_definitions(-DDATADIR=${DATA_ROOT})
message("Data root: ${DATA_ROOT}")

target_include_directories(yannpp_tests PRIVATE ${YANNPP_SOURCE_DIR})
target_include_directories(yannpp_tests PRIVATE ${MNIST_SOURCE_DIR})
target_include_directories(yannpp_tests PRIVATE ${GTEST_SOURCE_DIR}/googletest/include)
target_include_directories(yannpp_tests PRIVATE ${GTEST_SOURCE_DIR}/include)

# GTest needs threading support

target_link_libraries(yannpp_tests gtest_main)
target_link_libraries(yannpp_tests yannpp)

add_test(NAME YannppTests COMMAND yannpp_tests)
