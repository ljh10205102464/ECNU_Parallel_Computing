cmake_minimum_required(VERSION 3.6)

project(mnist_training CXX)

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

set (CMAKE_CXX_STANDARD 11)

set(SOURCES
    parsing/bmp_image.h
    parsing/bmp_image.cpp
    parsing/parsed_images.h
    parsing/parsed_images.cpp
    parsing/parsed_labels.h
    parsing/parsed_labels.cpp
    parsing/mnist_dataset.h
    parsing/mnist_dataset.cpp)

#add_executable(mnist_training ${SOURCES} mnist_simple.cpp)
#add_executable(mnist_training ${SOURCES} mnist_fc_network.cpp)
add_executable(mnist_training ${SOURCES} mnist_deeplearning.cpp)

target_include_directories(mnist_training PRIVATE ${YANNPP_SOURCE_DIR})
target_include_directories(mnist_training PRIVATE .)

target_link_libraries(mnist_training yannpp)
