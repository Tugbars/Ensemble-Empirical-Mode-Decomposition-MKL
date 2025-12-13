/**
 * EEMD-MKL Performance Benchmark
 *
 * Tests scaling across signal lengths, ensemble sizes, and thread counts.
 */

// Windows compatibility: M_PI is not standard C++
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "eemd_mkl.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace eemd;
using Clock = std::chrono::high_resolution_clock;

// Generate synthetic test signal
void generate_signal(double *signal, int32_t n, double dt)
{
    for (int32_t i = 0; i < n; ++i)
    {
        const double t = i * dt;
        const double trend = 2.0 * std::sin(2.0 * M_PI * 0.5 * t);
        const double mid = std::sin(2.0 * M_PI * 5.0 * t);
        const double high = 0.5 * std::sin(2.0 * M_PI * 25.0 * t);
        const double am = 1.0 + 0.3 * std::sin(2.0 * M_PI * 1.0 * t);
        signal[i] = trend + am * mid + high;
    }
}

struct BenchResult
{
    double time_ms;
    double throughput_msamples;
    int32_t n_imfs;
};

BenchResult run_benchmark(int32_t n, int32_t ensemble_size, int32_t n_trials)
{
    std::vector<double> signal(n);
    generate_signal(signal.data(), n, 0.01);

    EEMDConfig config;
    config.max_imfs = 8;
    config.ensemble_size = ensemble_size;
    config.noise_std = 0.2;

    EEMD decomposer(config);
    std::vector<std::vector<double>> imfs;
    int32_t n_imfs = 0;

    // Warmup
    decomposer.decompose(signal.data(), n, imfs, n_imfs);

    // Timed runs
    double total_time = 0.0;
    for (int32_t t = 0; t < n_trials; ++t)
    {
        auto t0 = Clock::now();
        decomposer.decompose(signal.data(), n, imfs, n_imfs);
        auto t1 = Clock::now();
        total_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    const double avg_time = total_time / n_trials;
    const double total_samples = static_cast<double>(ensemble_size) * n;
    const double throughput = (total_samples / 1e6) / (avg_time / 1000.0);

    return {avg_time, throughput, n_imfs};
}

void bench_signal_length()
{
    std::cout << "\n=== Signal Length Scaling ===\n";
    std::cout << "Ensemble: 100, Threads: " << omp_get_max_threads() << "\n\n";

    std::cout << std::setw(10) << "Length"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Throughput"
              << std::setw(8) << "IMFs" << "\n";
    std::cout << std::string(45, '-') << "\n";

    for (int32_t n : {256, 512, 1024, 2048, 4096, 8192})
    {
        auto r = run_benchmark(n, 100, 3);
        std::cout << std::setw(10) << n
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.time_ms
                  << std::setw(12) << std::setprecision(2) << r.throughput_msamples << " MS/s"
                  << std::setw(8) << r.n_imfs << "\n";
    }
}

void bench_ensemble_size()
{
    std::cout << "\n=== Ensemble Size Scaling ===\n";
    std::cout << "Signal: 1024, Threads: " << omp_get_max_threads() << "\n\n";

    std::cout << std::setw(10) << "Ensemble"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "ms/trial" << "\n";
    std::cout << std::string(49, '-') << "\n";

    for (int32_t ens : {10, 25, 50, 100, 200, 500})
    {
        auto r = run_benchmark(1024, ens, 3);
        std::cout << std::setw(10) << ens
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.time_ms
                  << std::setw(12) << std::setprecision(2) << r.throughput_msamples << " MS/s"
                  << std::setw(12) << std::setprecision(3) << r.time_ms / ens << "\n";
    }
}

void bench_thread_scaling()
{
    std::cout << "\n=== Thread Scaling ===\n";
    std::cout << "Signal: 1024, Ensemble: 100\n\n";

    const int32_t max_threads = omp_get_max_threads();

    std::cout << std::setw(8) << "Threads"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(47, '-') << "\n";

    double baseline_time = 0.0;

    for (int32_t threads = 1; threads <= max_threads; threads *= 2)
    {
        omp_set_num_threads(threads);
        mkl_set_num_threads(1); // CRITICAL: MKL sequential to avoid N² threads!

// Warm up thread pool
#pragma omp parallel
        {
            volatile int x = 0;
            (void)x;
        }

        auto r = run_benchmark(1024, 100, 5);

        if (threads == 1)
            baseline_time = r.time_ms;
        const double speedup = (r.time_ms > 0) ? baseline_time / r.time_ms : 0.0;

        std::cout << std::setw(8) << threads
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.time_ms
                  << std::setw(12) << std::setprecision(2) << r.throughput_msamples << " MS/s"
                  << std::setw(12) << std::setprecision(2) << speedup << "x";

        if (r.n_imfs == 0)
            std::cout << " [FAIL]";
        std::cout << "\n";
    }

    // Restore defaults
    omp_set_num_threads(max_threads);
    mkl_set_num_threads(1); // Keep MKL sequential for EEMD workloads
}

void bench_emd_vs_eemd()
{
    std::cout << "\n=== EMD vs EEMD Comparison ===\n";
    std::cout << "Signal: 2048\n\n";

    const int32_t n = 2048;
    std::vector<double> signal(n);
    generate_signal(signal.data(), n, 0.01);

    EEMDConfig config;
    config.max_imfs = 8;

    EEMD decomposer(config);

    // EMD (single decomposition)
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;

    auto t0 = Clock::now();
    for (int i = 0; i < 100; ++i)
    {
        decomposer.decompose_emd(signal.data(), n, imfs, residue);
    }
    auto t1 = Clock::now();
    double emd_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / 100.0;

    // EEMD (ensemble)
    config.ensemble_size = 100;
    EEMD eemd_decomposer(config);
    int32_t n_imfs;

    t0 = Clock::now();
    eemd_decomposer.decompose(signal.data(), n, imfs, n_imfs);
    t1 = Clock::now();
    double eemd_time = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Single EMD:        " << std::fixed << std::setprecision(2)
              << emd_time << " ms\n";
    std::cout << "EEMD (100 trials): " << eemd_time << " ms\n";
    std::cout << "Parallel efficiency: " << std::setprecision(1)
              << 100.0 * (emd_time * 100) / eemd_time << "%\n";
}

int main()
{
    // Initialize with low-latency settings
    eemd_init_low_latency(8, true); // Adjust core count for your CPU

    std::cout << "\nEEMD-MKL Performance Benchmark\n";
    std::cout << "==============================\n";
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";
    std::cout << "MKL threads: " << mkl_get_max_threads() << " (sequential for EEMD)\n";

    bench_signal_length();
    bench_ensemble_size();
    bench_thread_scaling();
    bench_emd_vs_eemd();

    std::cout << "\nBenchmark complete.\n";
    return 0;
}