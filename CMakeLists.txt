cmake_minimum_required(VERSION 3.20)

project(eemd_mkl 
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "EEMD/ICEEMDAN with Intel MKL"
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
# ICEEMDAN Interface Library (extends EEMD)
# ============================================================================

add_library(iceemdan_mkl INTERFACE)
add_library(eemd::iceemdan_mkl ALIAS iceemdan_mkl)

target_include_directories(iceemdan_mkl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# ICEEMDAN depends on EEMD
target_link_libraries(iceemdan_mkl INTERFACE eemd_mkl)

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

macro(add_iceemdan_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE iceemdan_mkl)
    if(USE_IPO)
        set_target_properties(${name} PROPERTIES 
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()
endmacro()

# ============================================================================
# Examples (in root)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/example_simple.cpp")
    add_iceemdan_executable(example_simple example_simple.cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/example_finance_csv.cpp")
    add_iceemdan_executable(example_finance_csv example_finance_csv.cpp)
endif()

# ============================================================================
# Tests and Benchmarks (in test/ subdirectory)
# ============================================================================

set(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")

# --- Core EEMD tests ---
if(EXISTS "${TEST_DIR}/eemd_bench.cpp")
    add_eemd_executable(eemd_bench ${TEST_DIR}/eemd_bench.cpp)
endif()

if(EXISTS "${TEST_DIR}/eemd_test.cpp")
    add_eemd_executable(eemd_test ${TEST_DIR}/eemd_test.cpp)
endif()

# --- ICEEMDAN tests ---
if(EXISTS "${TEST_DIR}/iceemdan_bench.cpp")
    add_iceemdan_executable(iceemdan_bench ${TEST_DIR}/iceemdan_bench.cpp)
endif()

if(EXISTS "${TEST_DIR}/iceemdan_accuracy_test.cpp")
    add_iceemdan_executable(accuracy_test ${TEST_DIR}/iceemdan_accuracy_test.cpp)
endif()

if(EXISTS "${TEST_DIR}/iceemdan_finance_bench.cpp")
    add_iceemdan_executable(iceemdan_finance_bench ${TEST_DIR}/iceemdan_finance_bench.cpp)
endif()

# --- Optimization benchmarks ---
if(EXISTS "${TEST_DIR}/bench_adaptive.cpp")
    add_iceemdan_executable(bench_adaptive ${TEST_DIR}/bench_adaptive.cpp)
endif()

if(EXISTS "${TEST_DIR}/bench_snumber.cpp")
    add_iceemdan_executable(bench_snumber ${TEST_DIR}/bench_snumber.cpp)
endif()

if(EXISTS "${TEST_DIR}/bench_simd.cpp")
    add_iceemdan_executable(bench_simd ${TEST_DIR}/bench_simd.cpp)
endif()

if(EXISTS "${TEST_DIR}/test_akima.cpp")
    add_iceemdan_executable(test_akima ${TEST_DIR}/test_akima.cpp)
endif()

if(EXISTS "${TEST_DIR}/spline_bench.cpp")
    add_eemd_executable(spline_bench ${TEST_DIR}/spline_bench.cpp)
endif()

# ============================================================================
# Installation (optional)
# ============================================================================

include(GNUInstallDirs)

install(TARGETS eemd_mkl iceemdan_mkl
    EXPORT eemd_mkl_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES 
    eemd_mkl.hpp 
    iceemdan_mkl.hpp
    iceemdan_mkl_opt.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT eemd_mkl_targets
    FILE eemd_mkl-targets.cmake
    NAMESPACE eemd::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/eemd_mkl
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== EEMD/ICEEMDAN-MKL Configuration ===")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  MKL interface: ${MKL_INTERFACE}")
message(STATUS "  MKL threading: ${MKL_THREADING}")
message(STATUS "  EEMD header: eemd_mkl.hpp")
message(STATUS "  ICEEMDAN header: iceemdan_mkl.hpp")
message(STATUS "")