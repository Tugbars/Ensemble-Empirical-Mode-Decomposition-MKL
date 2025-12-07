# EEMD-MKL

Fast Ensemble Empirical Mode Decomposition using Intel MKL.

## Features

- **Header-only** — single file `eemd_mkl.hpp`
- **MKL-accelerated** — cubic spline interpolation via MKL Data Fitting
- **Parallel** — OpenMP ensemble parallelization with thread-local accumulation
- **Zero-allocation hot path** — pre-allocated scratch buffers, grow-only memory
- **Portable SIMD** — compiler-aware `#pragma omp simd` (works with MSVC, ICX, GCC, Clang)

## Requirements

- Intel oneAPI MKL (2021+)
- C++17 compiler
- OpenMP support

## Quick Start

```cpp
#include "eemd_mkl.hpp"

int main() {
    // Initialize (call once at startup)
    init_14900kf(true);  // Configures threads, DAZ/FTZ, MKL settings
    
    // Configure
    eemd::EEMDConfig config;
    config.ensemble_size = 100;
    config.max_imfs = 10;
    config.noise_std = 0.2;
    
    // Decompose
    eemd::EEMD decomposer(config);
    
    std::vector<double> signal(1024);
    // ... fill signal ...
    
    std::vector<std::vector<double>> imfs;
    int32_t n_imfs;
    
    decomposer.decompose(signal.data(), signal.size(), imfs, n_imfs);
    
    // imfs[0..n_imfs-1] contains the IMF components
}
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_imfs` | 10 | Maximum IMFs to extract |
| `max_sift_iters` | 100 | Sifting iterations per IMF |
| `sift_threshold` | 0.05 | SD stopping criterion |
| `ensemble_size` | 100 | Number of noise trials |
| `noise_std` | 0.2 | Noise amplitude (fraction of signal std) |
| `boundary_extend` | 2 | Extrema to mirror at boundaries |
| `rng_seed` | 42 | Random seed for reproducibility |

**Speed tuning:**
```cpp
config.max_sift_iters = 50;   // Reduce iterations
config.sift_threshold = 0.1;  // Relax threshold
config.ensemble_size = 50;    // Fewer trials
```

## Build

### Windows (MSVC)
```batch
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### Windows (Intel ICX)
```batch
cmake .. -G "Visual Studio 17 2022" -T "Intel C++ Compiler 2025"
cmake --build . --config Release
```

### Linux
```bash
source /opt/intel/oneapi/setvars.sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Performance

Tested on Intel Core i9-14900KF (8 P-cores, MKL sequential, OpenMP parallel):

| Signal | Ensemble | Time | Throughput |
|--------|----------|------|------------|
| 256 | 100 | 7 ms | 3.5 MS/s |
| 1024 | 100 | 48 ms | 2.1 MS/s |
| 2048 | 100 | 92 ms | 2.2 MS/s |
| 8192 | 100 | 420 ms | 1.9 MS/s |

Single EMD latency: **~1 ms** for 1024 samples.

## API

### `eemd::EEMD`

```cpp
// EEMD decomposition (ensemble averaging)
bool decompose(
    const double* signal,
    int32_t n,
    std::vector<std::vector<double>>& imfs,
    int32_t& n_imfs
);

// Single EMD (no ensemble)
bool decompose_emd(
    const double* signal,
    int32_t n,
    std::vector<std::vector<double>>& imfs,
    std::vector<double>& residue
);
```

### `eemd::compute_instantaneous_frequency`

```cpp
bool compute_instantaneous_frequency(
    const double* imf,
    int32_t n,
    double* inst_freq,
    double sample_rate = 1.0
);
```

## Initialization Functions

| Function | Use Case |
|----------|----------|
| `init_14900kf()` | Low-latency (8 threads, infinite blocktime, P-cores only) |
| `init_14900kf_throughput()` | Batch processing (16 threads with HT) |

Adjust thread counts in the source for your specific CPU.

## License

MIT
