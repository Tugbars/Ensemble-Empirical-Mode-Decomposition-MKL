/**
 * EEMD-MKL Benchmark for Intel Core i9-14900KF
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include "eemd_mkl_14900kf.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace eemd;

// Generate test signal
void generate_signal(double* signal, int32_t n, double dt) {
    for (int32_t i = 0; i < n; ++i) {
        double t = i * dt;
        double trend = 2.0 * std::sin(2.0 * M_PI * 0.5 * t);
        double mid = std::sin(2.0 * M_PI * 5.0 * t);
        double high = 0.5 * std::sin(2.0 * M_PI * 25.0 * t);
        double am = 1.0 + 0.3 * std::sin(2.0 * M_PI * 1.0 * t);
        signal[i] = trend + am * mid + high;
    }
}

struct BenchResult {
    double time_ms;
    double throughput_msamples;
    int32_t n_imfs;
};

BenchResult run_benchmark(int32_t signal_len, int32_t ensemble_size, int trials) {
    std::vector<double> signal(signal_len);
    generate_signal(signal.data(), signal_len, 0.01);
    
    EEMDConfig config;
    config.ensemble_size = ensemble_size;
    config.max_imfs = 10;
    config.max_sift_iters = 50;
    
    EEMD_14900KF decomposer(config);
    std::vector<std::vector<double>> imfs;
    int32_t n_imfs = 0;
    
    // Warmup
    decomposer.decompose(signal.data(), signal_len, imfs, n_imfs);
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < trials; ++t) {
        decomposer.decompose(signal.data(), signal_len, imfs, n_imfs);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / trials;
    
    double total_samples = static_cast<double>(signal_len) * ensemble_size;
    double throughput = total_samples / (avg_ms * 1000.0);
    
    return {avg_ms, throughput, n_imfs};
}

void bench_signal_length() {
    std::cout << "\n=== Signal Length Scaling (14900KF) ===\n";
    std::cout << "Ensemble: 100, Threads: 8 P-cores\n\n";
    
    std::cout << std::setw(10) << "Length"
              << std::setw(14) << "Time (ms)"
              << std::setw(16) << "Throughput"
              << std::setw(8) << "IMFs" << "\n";
    std::cout << std::string(48, '-') << "\n";
    
    for (int32_t len : {256, 512, 1024, 2048, 4096, 8192}) {
        auto r = run_benchmark(len, 100, 5);
        
        std::cout << std::setw(10) << len
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.time_ms
                  << std::setw(12) << std::setprecision(2) << r.throughput_msamples << " MS/s"
                  << std::setw(8) << r.n_imfs << "\n";
    }
}

void bench_ensemble_size() {
    std::cout << "\n=== Ensemble Size Scaling (14900KF) ===\n";
    std::cout << "Signal: 1024, Threads: 8 P-cores\n\n";
    
    std::cout << std::setw(10) << "Ensemble"
              << std::setw(14) << "Time (ms)"
              << std::setw(16) << "Throughput"
              << std::setw(14) << "ms/trial" << "\n";
    std::cout << std::string(54, '-') << "\n";
    
    for (int32_t ens : {10, 25, 50, 100, 200, 500}) {
        auto r = run_benchmark(1024, ens, 3);
        
        std::cout << std::setw(10) << ens
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.time_ms
                  << std::setw(12) << std::setprecision(2) << r.throughput_msamples << " MS/s"
                  << std::setw(14) << std::setprecision(3) << (r.time_ms / ens) << "\n";
    }
}

void bench_single_emd() {
    std::cout << "\n=== Single EMD Latency (14900KF) ===\n\n";
    
    std::cout << std::setw(10) << "Length"
              << std::setw(14) << "Time (µs)"
              << std::setw(8) << "IMFs" << "\n";
    std::cout << std::string(32, '-') << "\n";
    
    for (int32_t len : {256, 512, 1024, 2048, 4096}) {
        std::vector<double> signal(len);
        generate_signal(signal.data(), len, 0.01);
        
        EEMDConfig config;
        config.max_sift_iters = 50;
        
        EEMD_14900KF decomposer(config);
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        
        // Warmup
        decomposer.decompose_emd(signal.data(), len, imfs, residue);
        
        // Timed
        const int trials = 100;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int t = 0; t < trials; ++t) {
            decomposer.decompose_emd(signal.data(), len, imfs, residue);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        double avg_us = elapsed_us / trials;
        
        std::cout << std::setw(10) << len
                  << std::setw(14) << std::fixed << std::setprecision(1) << avg_us
                  << std::setw(8) << imfs.size() << "\n";
    }
}

void bench_comparison() {
    std::cout << "\n=== EEMD vs EMD (14900KF) ===\n";
    std::cout << "Signal: 2048\n\n";
    
    std::vector<double> signal(2048);
    generate_signal(signal.data(), 2048, 0.01);
    
    EEMDConfig config;
    config.ensemble_size = 100;
    config.max_sift_iters = 50;
    
    EEMD_14900KF decomposer(config);
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    int32_t n_imfs;
    
    // Single EMD
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < 100; ++t) {
        decomposer.decompose_emd(signal.data(), 2048, imfs, residue);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double emd_ms = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;
    
    // EEMD
    start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < 10; ++t) {
        decomposer.decompose(signal.data(), 2048, imfs, n_imfs);
    }
    end = std::chrono::high_resolution_clock::now();
    double eemd_ms = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    
    double efficiency = (emd_ms * 100.0) / eemd_ms * 100.0;
    
    std::cout << "Single EMD:        " << std::fixed << std::setprecision(2) 
              << emd_ms << " ms\n";
    std::cout << "EEMD (100 trials): " << std::fixed << std::setprecision(2) 
              << eemd_ms << " ms\n";
    std::cout << "Parallel efficiency: " << std::fixed << std::setprecision(1)
              << efficiency << "%\n";
}

int main() {
    // Initialize for 14900KF
    init_14900kf(true);  // Verbose mode
    
    std::cout << "EEMD-MKL Benchmark: Intel Core i9-14900KF\n";
    std::cout << "==========================================\n";
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    std::cout << "MKL threads: " << mkl_get_max_threads() << " (sequential)\n";
    
    bench_signal_length();
    bench_ensemble_size();
    bench_single_emd();
    bench_comparison();
    
    std::cout << "\nBenchmark complete.\n";
    return 0;
}