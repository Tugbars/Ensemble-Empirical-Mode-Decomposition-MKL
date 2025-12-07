cmake_minimum_required(VERSION 3.20)

project(eemd_mkl 
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "EEMD with Intel MKL"
)

# C++17 minimum
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")

# ==============================================================================
# Intel MKL Configuration
# ==============================================================================

set(MKL_LINK "dynamic" CACHE STRING "MKL link type")
set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading")
set(MKL_INTERFACE "lp64" CACHE STRING "MKL interface")

find_package(MKL CONFIG REQUIRED)

message(STATUS "MKL found: ${MKL_ROOT}")

# ==============================================================================
# Compiler Flags
# ==============================================================================

if(MSVC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
    # Windows: MSVC or Intel icx-cl (MSVC-style flags)
    set(EEMD_CXX_FLAGS_RELEASE
        /O2
        /Ob2
        /Qopenmp
        /fp:fast
        /EHsc
        /DNDEBUG
    )
    
    set(EEMD_CXX_FLAGS_DEBUG
        /Od
        /Zi
        /Qopenmp
        /EHsc
        /D_DEBUG
    )
    
    # Try to use AVX2/AVX-512 if available
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        list(APPEND EEMD_CXX_FLAGS_RELEASE /QxHost)
    endif()
    
else()
    # Linux: Intel icpx or GCC (GNU-style flags)
    set(EEMD_CXX_FLAGS_RELEASE
        -O3
        -march=native
        -ffast-math
        -qopenmp
        -DNDEBUG
    )
    
    set(EEMD_CXX_FLAGS_DEBUG
        -O0
        -g
        -qopenmp
    )
endif()

# ==============================================================================
# Header-Only Library
# ==============================================================================

add_library(eemd_mkl INTERFACE)
add_library(eemd::eemd_mkl ALIAS eemd_mkl)

target_include_directories(eemd_mkl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(eemd_mkl INTERFACE MKL::MKL)

# ==============================================================================
# Executables
# ==============================================================================

# Test executable
add_executable(eemd_test eemd_test.cpp)
target_link_libraries(eemd_test PRIVATE eemd_mkl)
target_compile_options(eemd_test PRIVATE
    "$<$<CONFIG:Release>:${EEMD_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:Debug>:${EEMD_CXX_FLAGS_DEBUG}>"
)

# Benchmark executable
option(EEMD_BUILD_BENCH "Build benchmark" ON)
if(EEMD_BUILD_BENCH)
    add_executable(eemd_bench eemd_bench.cpp)
    target_link_libraries(eemd_bench PRIVATE eemd_mkl)
    target_compile_options(eemd_bench PRIVATE
        "$<$<CONFIG:Release>:${EEMD_CXX_FLAGS_RELEASE}>"
        "$<$<CONFIG:Debug>:${EEMD_CXX_FLAGS_DEBUG}>"
    )
endif()

# ==============================================================================
# Installation
# ==============================================================================

include(GNUInstallDirs)

install(FILES eemd_mkl.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS eemd_mkl
    EXPORT eemd_mklTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "")
message(STATUS "=== EEMD-MKL Configuration ===")
message(STATUS "  Compiler ID:     ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  MKL threading:   ${MKL_THREADING}")
message(STATUS "  MKL interface:   ${MKL_INTERFACE}")
message(STATUS "  MKL link:        ${MKL_LINK}")
message(STATUS "")