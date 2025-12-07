/**
 * EEMD Test and Example Usage
 * 
 * Compile:
 *   source /opt/intel/oneapi/setvars.sh  # or equivalent
 *   icpx -O3 -march=native -qopenmp -qmkl eemd_test.cpp -o eemd_test
 * 
 * Or with g++:
 *   g++ -O3 -march=native -fopenmp eemd_test.cpp -o eemd_test \
 *       -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 \
 *       -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm
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

using namespace eemd;

// Generate test signal: sum of sinusoids with different frequencies
void generate_test_signal(double* signal, int32_t n, double dt) {
    // Component 1: Low frequency trend
    // Component 2: Medium frequency oscillation  
    // Component 3: High frequency oscillation
    
    for (int32_t i = 0; i < n; ++i) {
        const double t = i * dt;
        
        // Slow trend (IMF should capture this last)
        const double trend = 2.0 * std::sin(2.0 * M_PI * 0.5 * t);
        
        // Medium frequency
        const double mid = 1.0 * std::sin(2.0 * M_PI * 5.0 * t);
        
        // High frequency  
        const double high = 0.5 * std::sin(2.0 * M_PI * 25.0 * t);
        
        // Add some AM modulation for realism
        const double am = 1.0 + 0.3 * std::sin(2.0 * M_PI * 1.0 * t);
        
        signal[i] = trend + am * mid + high;
    }
}

// Generate chirp signal (frequency sweep)
void generate_chirp_signal(double* signal, int32_t n, double dt) {
    const double f0 = 1.0;   // Start frequency
    const double f1 = 50.0;  // End frequency
    const double T = n * dt;
    
    for (int32_t i = 0; i < n; ++i) {
        const double t = i * dt;
        const double phase = 2.0 * M_PI * (f0 * t + (f1 - f0) * t * t / (2.0 * T));
        signal[i] = std::sin(phase);
    }
}

// Compute energy of a signal
double compute_energy(const double* signal, int32_t n) {
    double e = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        e += signal[i] * signal[i];
    }
    return e;
}

// Compute correlation between two signals
double compute_correlation(const double* a, const double* b, int32_t n) {
    double sum_a = 0.0, sum_b = 0.0, sum_ab = 0.0;
    double sum_a2 = 0.0, sum_b2 = 0.0;
    
    for (int32_t i = 0; i < n; ++i) {
        sum_a += a[i];
        sum_b += b[i];
        sum_ab += a[i] * b[i];
        sum_a2 += a[i] * a[i];
        sum_b2 += b[i] * b[i];
    }
    
    const double mean_a = sum_a / n;
    const double mean_b = sum_b / n;
    const double var_a = sum_a2 / n - mean_a * mean_a;
    const double var_b = sum_b2 / n - mean_b * mean_b;
    const double cov = sum_ab / n - mean_a * mean_b;
    
    if (var_a < 1e-15 || var_b < 1e-15) return 0.0;
    
    return cov / std::sqrt(var_a * var_b);
}

void test_basic_emd() {
    std::cout << "=== Basic EMD Test ===" << std::endl;
    
    const int32_t n = 1024;
    const double dt = 0.01;  // 100 Hz sampling
    
    std::vector<double> signal(n);
    generate_test_signal(signal.data(), n, dt);
    
    EEMDConfig config;
    config.max_imfs = 8;
    config.max_sift_iters = 100;
    config.sift_threshold = 0.05;
    
    EEMD decomposer(config);
    
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    bool success = decomposer.decompose_emd(signal.data(), n, imfs, residue);
    auto t_end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    if (!success) {
        std::cerr << "EMD decomposition failed!" << std::endl;
        return;
    }
    
    std::cout << "Decomposition time: " << std::fixed << std::setprecision(2) 
              << elapsed << " ms" << std::endl;
    std::cout << "Number of IMFs: " << imfs.size() << std::endl;
    
    // Energy analysis
    const double original_energy = compute_energy(signal.data(), n);
    std::cout << "\nEnergy distribution:" << std::endl;
    std::cout << "  Original signal: " << std::scientific << std::setprecision(4) 
              << original_energy << std::endl;
    
    double total_imf_energy = 0.0;
    for (size_t k = 0; k < imfs.size(); ++k) {
        const double e = compute_energy(imfs[k].data(), n);
        total_imf_energy += e;
        std::cout << "  IMF " << k + 1 << ": " << e 
                  << " (" << std::fixed << std::setprecision(1) 
                  << 100.0 * e / original_energy << "%)" << std::endl;
    }
    
    const double residue_energy = compute_energy(residue.data(), n);
    total_imf_energy += residue_energy;
    std::cout << "  Residue: " << std::scientific << residue_energy 
              << " (" << std::fixed << std::setprecision(1)
              << 100.0 * residue_energy / original_energy << "%)" << std::endl;
    
    // Reconstruction error
    std::vector<double> reconstructed(n, 0.0);
    for (size_t k = 0; k < imfs.size(); ++k) {
        for (int32_t i = 0; i < n; ++i) {
            reconstructed[i] += imfs[k][i];
        }
    }
    for (int32_t i = 0; i < n; ++i) {
        reconstructed[i] += residue[i];
    }
    
    double reconstruction_error = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        const double d = signal[i] - reconstructed[i];
        reconstruction_error += d * d;
    }
    reconstruction_error = std::sqrt(reconstruction_error / n);
    
    std::cout << "\nReconstruction RMSE: " << std::scientific << std::setprecision(6) 
              << reconstruction_error << std::endl;
}

void test_eemd_ensemble() {
    std::cout << "\n=== EEMD Ensemble Test ===" << std::endl;
    
    const int32_t n = 512;
    const double dt = 0.01;
    
    std::vector<double> signal(n);
    generate_test_signal(signal.data(), n, dt);
    
    EEMDConfig config;
    config.max_imfs = 6;
    config.ensemble_size = 50;  // Reduced for faster testing
    config.noise_std = 0.2;
    config.rng_seed = 12345;
    
    EEMD decomposer(config);
    
    std::vector<std::vector<double>> imfs;
    int32_t n_imfs = 0;
    
    std::cout << "Running EEMD with " << config.ensemble_size << " ensemble members..." << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    bool success = decomposer.decompose(signal.data(), n, imfs, n_imfs);
    auto t_end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    if (!success) {
        std::cerr << "EEMD decomposition failed!" << std::endl;
        return;
    }
    
    std::cout << "Decomposition time: " << std::fixed << std::setprecision(2) 
              << elapsed << " ms" << std::endl;
    std::cout << "Number of IMFs: " << n_imfs << std::endl;
    std::cout << "Throughput: " << std::setprecision(1) 
              << (config.ensemble_size * n / 1000.0) / (elapsed / 1000.0) 
              << " ksamples/sec" << std::endl;
    
    // Orthogonality index (IMFs should be approximately orthogonal)
    std::cout << "\nIMF correlations (should be near 0 for good decomposition):" << std::endl;
    for (size_t i = 0; i < imfs.size(); ++i) {
        for (size_t j = i + 1; j < imfs.size(); ++j) {
            double corr = compute_correlation(imfs[i].data(), imfs[j].data(), n);
            if (std::abs(corr) > 0.1) {  // Only show significant correlations
                std::cout << "  corr(IMF" << i + 1 << ", IMF" << j + 1 << ") = "
                          << std::fixed << std::setprecision(3) << corr << std::endl;
            }
        }
    }
}

void test_chirp_signal() {
    std::cout << "\n=== Chirp Signal Test ===" << std::endl;
    
    const int32_t n = 2048;
    const double dt = 0.001;  // 1000 Hz sampling
    
    std::vector<double> signal(n);
    generate_chirp_signal(signal.data(), n, dt);
    
    EEMDConfig config;
    config.max_imfs = 4;
    config.ensemble_size = 20;
    config.noise_std = 0.1;
    
    EEMD decomposer(config);
    
    std::vector<std::vector<double>> imfs;
    int32_t n_imfs = 0;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    bool success = decomposer.decompose(signal.data(), n, imfs, n_imfs);
    auto t_end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    if (!success) {
        std::cerr << "Chirp decomposition failed!" << std::endl;
        return;
    }
    
    std::cout << "Decomposition time: " << std::fixed << std::setprecision(2) 
              << elapsed << " ms" << std::endl;
    std::cout << "Number of IMFs: " << n_imfs << std::endl;
    
    // For chirp, most energy should be in first IMF
    const double original_energy = compute_energy(signal.data(), n);
    
    for (int32_t k = 0; k < n_imfs; ++k) {
        const double e = compute_energy(imfs[k].data(), n);
        std::cout << "  IMF " << k + 1 << " energy fraction: " 
                  << std::fixed << std::setprecision(1) 
                  << 100.0 * e / original_energy << "%" << std::endl;
    }
}

void test_instantaneous_frequency() {
    std::cout << "\n=== Instantaneous Frequency Test ===" << std::endl;
    
    const int32_t n = 512;
    const double dt = 0.01;
    const double fs = 1.0 / dt;
    const double f_test = 10.0;  // 10 Hz sinusoid
    
    std::vector<double> signal(n);
    for (int32_t i = 0; i < n; ++i) {
        signal[i] = std::sin(2.0 * M_PI * f_test * i * dt);
    }
    
    std::vector<double> inst_freq(n);
    
    bool success = compute_instantaneous_frequency(signal.data(), n, inst_freq.data(), fs);
    
    if (!success) {
        std::cerr << "Instantaneous frequency computation failed!" << std::endl;
        return;
    }
    
    // Compute mean IF (should be close to f_test)
    double mean_if = 0.0;
    int32_t count = 0;
    for (int32_t i = n / 4; i < 3 * n / 4; ++i) {  // Use middle portion
        mean_if += inst_freq[i];
        ++count;
    }
    mean_if /= count;
    
    std::cout << "Expected frequency: " << f_test << " Hz" << std::endl;
    std::cout << "Estimated mean IF:  " << std::fixed << std::setprecision(2) 
              << mean_if << " Hz" << std::endl;
    std::cout << "Relative error:     " << std::setprecision(2) 
              << 100.0 * std::abs(mean_if - f_test) / f_test << "%" << std::endl;
}

void benchmark() {
    std::cout << "\n=== Performance Benchmark ===" << std::endl;
    
    const std::vector<int32_t> sizes = {256, 512, 1024, 2048, 4096};
    const int32_t ensemble_size = 100;
    const int32_t n_trials = 3;
    
    EEMDConfig config;
    config.max_imfs = 6;
    config.ensemble_size = ensemble_size;
    config.noise_std = 0.2;
    
    std::cout << "Ensemble size: " << ensemble_size << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "\nSignal Length | Time (ms) | Throughput (Msamples/s)" << std::endl;
    std::cout << "--------------|-----------|------------------------" << std::endl;
    
    for (int32_t n : sizes) {
        std::vector<double> signal(n);
        generate_test_signal(signal.data(), n, 0.01);
        
        EEMD decomposer(config);
        std::vector<std::vector<double>> imfs;
        int32_t n_imfs = 0;
        
        double total_time = 0.0;
        
        for (int32_t trial = 0; trial < n_trials; ++trial) {
            auto t_start = std::chrono::high_resolution_clock::now();
            decomposer.decompose(signal.data(), n, imfs, n_imfs);
            auto t_end = std::chrono::high_resolution_clock::now();
            
            total_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();
        }
        
        const double avg_time = total_time / n_trials;
        const double throughput = (static_cast<double>(ensemble_size) * n / 1e6) / (avg_time / 1000.0);
        
        std::cout << std::setw(13) << n << " | "
                  << std::setw(9) << std::fixed << std::setprecision(1) << avg_time << " | "
                  << std::setw(22) << std::setprecision(2) << throughput << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Initialize with low-latency settings
    eemd_init_low_latency(8);  // Adjust core count for your CPU
    
    std::cout << "EEMD-MKL Test Suite" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
    std::cout << "MKL threads: " << mkl_get_max_threads() << " (sequential)" << std::endl;
    
    test_basic_emd();
    test_eemd_ensemble();
    test_chirp_signal();
    test_instantaneous_frequency();
    
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        benchmark();
    }
    
    std::cout << "\nAll tests completed." << std::endl;
    
    return 0;
}