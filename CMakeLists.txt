cmake_minimum_required(VERSION 3.20)

project(eemd_mkl 
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "EEMD with Intel MKL"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# MKL Configuration (The "Working" Pattern)
# ============================================================================

# CRITICAL: Force LP64 (32-bit integers) to fix the -1025 Error
set(MKL_INTERFACE "lp64" CACHE STRING "MKL interface (lp64 or ilp64)" FORCE)
set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading" FORCE)

# Find MKL using Intel's Config script
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
    # Intel ICX/ICPX
    message(STATUS "Detected Intel LLVM compiler (ICX)")
    if(WIN32)
        set(CXX_FLAGS_RELEASE
            /O3 /QxHost /Qipo /fp:fast /Qopenmp /EHsc /DNDEBUG
        )
    else()
        set(CXX_FLAGS_RELEASE
            -O3 -xHost -ipo -fp-model fast -qopenmp -DNDEBUG
        )
    endif()
    # Add IPO property
    set(USE_IPO TRUE)

elseif(MSVC)
    # Microsoft Visual C++
    message(STATUS "Detected MSVC compiler")
    set(CXX_FLAGS_RELEASE
        /O2 /Ob2 /fp:fast /arch:AVX2 /GL /EHsc /DNDEBUG /openmp
    )
    set(USE_IPO TRUE) # /GL implies LTCG

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang
    message(STATUS "Detected GCC/Clang compiler")
    set(CXX_FLAGS_RELEASE
        -O3 -march=native -ffast-math -funroll-loops -flto -fopenmp -DNDEBUG
    )
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

# Apply compiler flags to the interface
target_compile_options(eemd_mkl INTERFACE 
    $<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>
    $<$<CONFIG:Debug>:-g -O0>
)

# Link MKL and OpenMP
# Note: MKL::MKL is the meta-target that reads MKL_INTERFACE variable
target_link_libraries(eemd_mkl INTERFACE 
    MKL::MKL 
    OpenMP::OpenMP_CXX
)

# Define macros
target_compile_definitions(eemd_mkl INTERFACE 
    MKL_LP64 
    _USE_MATH_DEFINES
)

# ============================================================================
# Executables
# ============================================================================

# Helper macro
macro(add_eemd_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE eemd_mkl)
    
    # Enable Interprocedural Optimization if supported
    if(USE_IPO)
        set_target_properties(${name} PROPERTIES 
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()
endmacro()

# Benchmark
add_eemd_executable(eemd_bench eemd_bench.cpp)

# Debug Tool (Optional)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/eemd_debug.cpp")
    add_eemd_executable(eemd_debug eemd_debug.cpp)
endif()

# ============================================================================
# Build Instructions
# ============================================================================
message(STATUS "")
message(STATUS "=== Build Instructions ===")
message(STATUS "  1. mkdir build && cd build")
if(WIN32)
    message(STATUS "  2. cmake -G Ninja -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release ..")
else()
    message(STATUS "  2. cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release ..")
endif()
message(STATUS "  3. cmake --build . --config Release")
message(STATUS "")