cmake_minimum_required(VERSION 3.9)

project(yannpp CXX)

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
    common/cpphelpers.h
    common/cpphelpers.cpp
    common/shape.h
    common/array3d.h
    common/array3d_math.h
    common/log.h
    common/log.cpp
    common/utils.h
    common/utils.cpp
    optimizer/sdg_optimizer.h
    optimizer/optimizer.h
    network/network2.h
#    network/network1.h
#    network/network1.cpp
    layers/fullyconnectedlayer.h
    layers/poolinglayer.h
    layers/crossentropyoutputlayer.h
    layers/convolutionlayer.h
    layers/layer_base.h
    layers/layer_metadata.h
    network/activator.h)

add_library(yannpp SHARED STATIC ${SOURCES})

target_include_directories(yannpp PRIVATE ${YANNPP_SOURCE_DIR})

install(TARGETS yannpp DESTINATION lib)
