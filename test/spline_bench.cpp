/**
 * Spline Benchmark: MKL vs Hand-Written
 * 
 * Tests cubic spline performance for typical EEMD extrema counts
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include "cubic_spline_avx2.hpp"
#include <mkl.h>
#include <mkl_df.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace eemd;

// Generate random knot points (simulating extrema)
void generate_knots(double* x, double* y, int32_t n, int32_t signal_len) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // Generate sorted x values spanning [0, signal_len]
    std::vector<double> raw_x(n);
    for (int32_t i = 0; i < n; ++i) {
        raw_x[i] = dist(rng);
    }
    std::sort(raw_x.begin(), raw_x.end());
    
    for (int32_t i = 0; i < n; ++i) {
        x[i] = raw_x[i] * (signal_len - 1);
        y[i] = std::sin(2.0 * M_PI * x[i] / signal_len) + 0.5 * dist(rng);
    }
    
    // Ensure coverage of endpoints
    x[0] = -1.0;
    x[n-1] = static_cast<double>(signal_len);
}

// MKL spline wrapper for benchmarking
class MKLSplineBench {
public:
    MKLSplineBench(int32_t max_knots) {
        coeffs_.resize(4 * max_knots);
    }
    
    ~MKLSplineBench() {
        if (task_) dfDeleteTask(&task_);
    }
    
    bool construct(const double* x, const double* y, int32_t n) {
        if (task_) {
            dfDeleteTask(&task_);
            task_ = nullptr;
        }
        
        MKL_INT status = dfdNewTask1D(&task_, n, x, DF_NON_UNIFORM_PARTITION, 1, y, DF_NO_HINT);
        if (status != DF_STATUS_OK) return false;
        
        status = dfdEditPPSpline1D(task_, DF_PP_CUBIC, DF_PP_NATURAL,
                                    DF_BC_FREE_END, nullptr, DF_NO_IC, nullptr,
                                    coeffs_.data(), DF_NO_HINT);
        if (status != DF_STATUS_OK) return false;
        
        status = dfdConstruct1D(task_, DF_PP_SPLINE, DF_METHOD_STD);
        return (status == DF_STATUS_OK);
    }
    
    bool evaluate(const double* sites, double* results, int32_t n_sites) {
        const MKL_INT dorder[] = {1};
        MKL_INT status = dfdInterpolate1D(task_, DF_INTERP, DF_METHOD_PP,
                                           n_sites, sites, DF_SORTED_DATA,
                                           1, dorder, nullptr,
                                           results, DF_NO_HINT, nullptr);
        return (status == DF_STATUS_OK);
    }
    
private:
    DFTaskPtr task_ = nullptr;
    std::vector<double> coeffs_;
};

struct BenchResult {
    double construct_us;
    double evaluate_us;
    double total_us;
};

// Benchmark MKL spline
BenchResult bench_mkl(int32_t n_knots, int32_t n_sites, int trials) {
    std::vector<double> x(n_knots), y(n_knots);
    generate_knots(x.data(), y.data(), n_knots, n_sites);
    
    std::vector<double> sites(n_sites), results(n_sites);
    for (int32_t i = 0; i < n_sites; ++i) sites[i] = static_cast<double>(i);
    
    MKLSplineBench spline(n_knots);
    
    // Warmup
    spline.construct(x.data(), y.data(), n_knots);
    spline.evaluate(sites.data(), results.data(), n_sites);
    
    // Benchmark construct
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; ++t) {
        spline.construct(x.data(), y.data(), n_knots);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double construct_us = std::chrono::duration<double, std::micro>(end - start).count() / trials;
    
    // Benchmark evaluate
    start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; ++t) {
        spline.evaluate(sites.data(), results.data(), n_sites);
    }
    end = std::chrono::high_resolution_clock::now();
    double evaluate_us = std::chrono::duration<double, std::micro>(end - start).count() / trials;
    
    return {construct_us, evaluate_us, construct_us + evaluate_us};
}

// Benchmark hand-written spline
BenchResult bench_fast(int32_t n_knots, int32_t n_sites, int trials) {
    std::vector<double> x(n_knots), y(n_knots);
    generate_knots(x.data(), y.data(), n_knots, n_sites);
    
    std::vector<double> sites(n_sites), results(n_sites);
    for (int32_t i = 0; i < n_sites; ++i) sites[i] = static_cast<double>(i);
    
    FastCubicSpline spline(n_knots);
    
    // Warmup
    spline.construct(x.data(), y.data(), n_knots);
    spline.evaluate(sites.data(), results.data(), n_sites);
    
    // Benchmark construct
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; ++t) {
        spline.construct(x.data(), y.data(), n_knots);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double construct_us = std::chrono::duration<double, std::micro>(end - start).count() / trials;
    
    // Benchmark evaluate
    start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; ++t) {
        spline.evaluate(sites.data(), results.data(), n_sites);
    }
    end = std::chrono::high_resolution_clock::now();
    double evaluate_us = std::chrono::duration<double, std::micro>(end - start).count() / trials;
    
    return {construct_us, evaluate_us, construct_us + evaluate_us};
}

// Verify correctness
bool verify_correctness(int32_t n_knots, int32_t n_sites) {
    std::vector<double> x(n_knots), y(n_knots);
    generate_knots(x.data(), y.data(), n_knots, n_sites);
    
    std::vector<double> sites(n_sites);
    for (int32_t i = 0; i < n_sites; ++i) sites[i] = static_cast<double>(i);
    
    std::vector<double> mkl_results(n_sites), fast_results(n_sites);
    
    // MKL
    MKLSplineBench mkl_spline(n_knots);
    mkl_spline.construct(x.data(), y.data(), n_knots);
    mkl_spline.evaluate(sites.data(), mkl_results.data(), n_sites);
    
    // Fast
    FastCubicSpline fast_spline(n_knots);
    fast_spline.construct(x.data(), y.data(), n_knots);
    fast_spline.evaluate(sites.data(), fast_results.data(), n_sites);
    
    // Compare
    double max_error = 0.0;
    double max_rel_error = 0.0;
    
    for (int32_t i = 0; i < n_sites; ++i) {
        double err = std::abs(mkl_results[i] - fast_results[i]);
        double rel = (std::abs(mkl_results[i]) > 1e-10) 
                     ? err / std::abs(mkl_results[i]) 
                     : err;
        max_error = std::max(max_error, err);
        max_rel_error = std::max(max_rel_error, rel);
    }
    
    std::cout << "Correctness check (n_knots=" << n_knots << ", n_sites=" << n_sites << "):\n";
    std::cout << "  Max absolute error: " << std::scientific << max_error << "\n";
    std::cout << "  Max relative error: " << std::scientific << max_rel_error << "\n";
    std::cout << "  Status: " << (max_rel_error < 1e-10 ? "PASS" : "FAIL") << "\n\n";
    
    return max_rel_error < 1e-10;
}

int main() {
    std::cout << "Spline Benchmark: MKL vs Hand-Written AVX2\n";
    std::cout << "==========================================\n\n";
    
    // Verify correctness first
    std::cout << "=== Correctness Verification ===\n";
    verify_correctness(10, 256);
    verify_correctness(20, 512);
    verify_correctness(50, 1024);
    
    // Benchmark varying knot count (typical EEMD extrema counts)
    std::cout << "=== Knot Count Scaling (Signal: 1024) ===\n";
    std::cout << "Typical EEMD has 10-50 extrema per envelope\n\n";
    
    std::cout << std::setw(8) << "Knots"
              << std::setw(14) << "MKL (µs)"
              << std::setw(14) << "Fast (µs)"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(48, '-') << "\n";
    
    for (int32_t knots : {5, 10, 15, 20, 30, 50, 100}) {
        auto mkl = bench_mkl(knots, 1024, 1000);
        auto fast = bench_fast(knots, 1024, 1000);
        double speedup = mkl.total_us / fast.total_us;
        
        std::cout << std::setw(8) << knots
                  << std::setw(14) << std::fixed << std::setprecision(2) << mkl.total_us
                  << std::setw(14) << std::fixed << std::setprecision(2) << fast.total_us
                  << std::setw(11) << std::setprecision(1) << speedup << "x\n";
    }
    
    // Benchmark varying signal length
    std::cout << "\n=== Signal Length Scaling (Knots: 20) ===\n\n";
    
    std::cout << std::setw(10) << "Signal"
              << std::setw(14) << "MKL (µs)"
              << std::setw(14) << "Fast (µs)"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(50, '-') << "\n";
    
    for (int32_t signal : {256, 512, 1024, 2048, 4096, 8192}) {
        auto mkl = bench_mkl(20, signal, 500);
        auto fast = bench_fast(20, signal, 500);
        double speedup = mkl.total_us / fast.total_us;
        
        std::cout << std::setw(10) << signal
                  << std::setw(14) << std::fixed << std::setprecision(2) << mkl.total_us
                  << std::setw(14) << std::fixed << std::setprecision(2) << fast.total_us
                  << std::setw(11) << std::setprecision(1) << speedup << "x\n";
    }
    
    // Detailed breakdown
    std::cout << "\n=== Detailed Breakdown (20 knots, 1024 sites) ===\n\n";
    
    auto mkl = bench_mkl(20, 1024, 1000);
    auto fast = bench_fast(20, 1024, 1000);
    
    std::cout << std::setw(20) << ""
              << std::setw(14) << "MKL (µs)"
              << std::setw(14) << "Fast (µs)"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    std::cout << std::setw(20) << "Construct"
              << std::setw(14) << std::fixed << std::setprecision(2) << mkl.construct_us
              << std::setw(14) << std::fixed << std::setprecision(2) << fast.construct_us
              << std::setw(11) << std::setprecision(1) << mkl.construct_us / fast.construct_us << "x\n";
    
    std::cout << std::setw(20) << "Evaluate"
              << std::setw(14) << std::fixed << std::setprecision(2) << mkl.evaluate_us
              << std::setw(14) << std::fixed << std::setprecision(2) << fast.evaluate_us
              << std::setw(11) << std::setprecision(1) << mkl.evaluate_us / fast.evaluate_us << "x\n";
    
    std::cout << std::setw(20) << "Total"
              << std::setw(14) << std::fixed << std::setprecision(2) << mkl.total_us
              << std::setw(14) << std::fixed << std::setprecision(2) << fast.total_us
              << std::setw(11) << std::setprecision(1) << mkl.total_us / fast.total_us << "x\n";
    
    // EEMD workload estimate
    std::cout << "\n=== EEMD Workload Estimate ===\n";
    std::cout << "Typical: 100 ensembles × 8 IMFs × 20 sift iters × 2 envelopes\n";
    std::cout << "         = 32,000 spline construct+evaluate cycles\n\n";
    
    double mkl_eemd_ms = mkl.total_us * 32000 / 1000.0;
    double fast_eemd_ms = fast.total_us * 32000 / 1000.0;
    
    std::cout << "MKL:  " << std::fixed << std::setprecision(1) << mkl_eemd_ms << " ms\n";
    std::cout << "Fast: " << std::fixed << std::setprecision(1) << fast_eemd_ms << " ms\n";
    std::cout << "Savings: " << std::fixed << std::setprecision(1) 
              << (mkl_eemd_ms - fast_eemd_ms) << " ms ("
              << std::setprecision(0) << (1.0 - fast_eemd_ms/mkl_eemd_ms) * 100 << "% reduction)\n";
    
    std::cout << "\nBenchmark complete.\n";
    return 0;
}
