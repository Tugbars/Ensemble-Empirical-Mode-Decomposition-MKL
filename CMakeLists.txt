cmake_minimum_required(VERSION 3.20)

project(eemd_mkl 
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "EEMD with Intel MKL"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# MKL Configuration
# ============================================================================

# CRITICAL: Force LP64 (32-bit integers)
set(MKL_INTERFACE "lp64" CACHE STRING "MKL interface (lp64 or ilp64)" FORCE)
set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading" FORCE)

find_package(MKL CONFIG REQUIRED)

message(STATUS "MKL found: ${MKL_ROOT}")
message(STATUS "MKL interface: ${MKL_INTERFACE}")

# ============================================================================
# OpenMP Support
# ============================================================================
find_package(OpenMP REQUIRED)

# ============================================================================
# Compiler Detection & Flags
# ============================================================================

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(STATUS "Detected Intel LLVM compiler (ICX)")
    if(WIN32)
        set(CXX_FLAGS_RELEASE /O3 /QxHost /Qipo /fp:fast /Qopenmp /EHsc /DNDEBUG)
    else()
        set(CXX_FLAGS_RELEASE -O3 -xHost -ipo -fp-model fast -qopenmp -DNDEBUG)
    endif()
    set(USE_IPO TRUE)

elseif(MSVC)
    message(STATUS "Detected MSVC compiler")
    set(CXX_FLAGS_RELEASE /O2 /Ob2 /fp:fast /arch:AVX2 /GL /EHsc /DNDEBUG /openmp)
    set(USE_IPO TRUE)

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Detected GCC/Clang compiler")
    set(CXX_FLAGS_RELEASE -O3 -march=native -ffast-math -funroll-loops -flto -fopenmp -DNDEBUG)
    set(USE_IPO TRUE)
endif()

# ============================================================================
# Main Interface Library (Header Only)
# ============================================================================

add_library(eemd_mkl INTERFACE)
add_library(eemd::eemd_mkl ALIAS eemd_mkl)

target_include_directories(eemd_mkl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(eemd_mkl INTERFACE 
    $<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>
)

target_link_libraries(eemd_mkl INTERFACE 
    MKL::MKL 
    OpenMP::OpenMP_CXX
)

target_compile_definitions(eemd_mkl INTERFACE 
    MKL_LP64 
    _USE_MATH_DEFINES
)

# ============================================================================
# Executables
# ============================================================================

macro(add_eemd_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE eemd_mkl)
    if(USE_IPO)
        set_target_properties(${name} PROPERTIES 
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()
endmacro()

add_eemd_executable(eemd_bench eemd_bench.cpp)
add_eemd_executable(eemd_test eemd_test.cpp)

# 14900KF optimized benchmark
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/eemd_bench_14900kf.cpp")
    add_eemd_executable(eemd_bench_14900kf eemd_bench_14900kf.cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/eemd_debug.cpp")
    add_eemd_executable(eemd_debug eemd_debug.cpp)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== EEMD-MKL Configuration ===")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  MKL interface: ${MKL_INTERFACE}")
message(STATUS "  MKL threading: ${MKL_THREADING}")
message(STATUS "")