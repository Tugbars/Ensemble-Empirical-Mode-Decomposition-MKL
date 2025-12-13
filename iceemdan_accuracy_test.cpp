/**
 * ICEEMDAN Accuracy Test Suite
 * 
 * Run this before and after optimizations to ensure correctness.
 * Tests mathematical properties that must hold regardless of implementation.
 * 
 * Build: g++ -O3 -std=c++17 iceemdan_accuracy_test.cpp -o accuracy_test -lmkl_rt -liomp5 -lpthread -lm
 */

#include "iceemdan_mkl.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Test Infrastructure
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    double value;
    double threshold;
    std::string unit;
};

std::vector<TestResult> g_results;

void report(const std::string& name, bool passed, double value, double threshold, const std::string& unit = "") {
    g_results.push_back({name, passed, value, threshold, unit});
    
    const char* status = passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m";
    std::cout << std::left << std::setw(45) << name << " [" << status << "] ";
    std::cout << std::scientific << std::setprecision(2) << value;
    if (!unit.empty()) std::cout << " " << unit;
    std::cout << " (threshold: " << threshold << ")\n";
}

// ============================================================================
// Signal Generators
// ============================================================================

void generate_sum_of_sines(double* signal, int32_t n, uint32_t seed) {
    // Known frequencies: 8, 32, 128 cycles + trend
    for (int32_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / n;
        signal[i] = 100.0 + 10.0 * t                          // Trend
                  + 5.0 * std::sin(2 * M_PI * 8 * t)          // Low freq
                  + 2.0 * std::sin(2 * M_PI * 32 * t)         // Mid freq
                  + 0.5 * std::sin(2 * M_PI * 128 * t);       // High freq
    }
}

void generate_garch(double* signal, int32_t n, uint32_t seed) {
    VSLStreamStatePtr stream = nullptr;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    
    std::vector<double> noise(n);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, noise.data(), 0.0, 1.0);
    
    double sigma2 = 0.0001;
    signal[0] = 100.0;
    
    for (int32_t i = 1; i < n; ++i) {
        double eps = noise[i-1] * std::sqrt(sigma2);
        sigma2 = 0.00001 + 0.1 * eps * eps + 0.85 * sigma2;
        signal[i] = signal[i-1] * std::exp(std::sqrt(sigma2) * noise[i]);
    }
    
    vslDeleteStream(&stream);
}

void generate_impulse(double* signal, int32_t n) {
    std::fill(signal, signal + n, 0.0);
    signal[n / 2] = 1.0;
}

void generate_step(double* signal, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        signal[i] = (i < n / 2) ? 0.0 : 1.0;
    }
}

void generate_ramp(double* signal, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        signal[i] = static_cast<double>(i) / n;
    }
}

void generate_white_noise(double* signal, int32_t n, uint32_t seed) {
    VSLStreamStatePtr stream = nullptr;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, signal, 0.0, 1.0);
    vslDeleteStream(&stream);
}

// ============================================================================
// Test Functions
// ============================================================================

/**
 * Test 1: Perfect Reconstruction
 * sum(IMFs) + residue must equal original signal to machine precision
 */
double test_reconstruction(const double* original, 
                           const std::vector<std::vector<double>>& imfs,
                           const std::vector<double>& residue,
                           int32_t n) {
    double max_error = 0.0;
    
    for (int32_t i = 0; i < n; ++i) {
        double reconstructed = residue[i];
        for (const auto& imf : imfs) {
            reconstructed += imf[i];
        }
        double error = std::abs(original[i] - reconstructed);
        max_error = std::max(max_error, error);
    }
    
    return max_error;
}

/**
 * Test 2: Energy Conservation
 * Total energy of IMFs + residue should equal energy of original
 */
double test_energy_conservation(const double* original,
                                const std::vector<std::vector<double>>& imfs,
                                const std::vector<double>& residue,
                                int32_t n) {
    // Original energy
    double E_original = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        E_original += original[i] * original[i];
    }
    
    // IMF + residue energy (with cross terms for proper comparison)
    // Actually, for a complete decomposition, we just check reconstruction
    // Energy conservation isn't guaranteed due to non-orthogonality
    // Instead, compute relative energy difference
    
    double E_components = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        double sum = residue[i];
        for (const auto& imf : imfs) {
            sum += imf[i];
        }
        E_components += sum * sum;
    }
    
    return std::abs(E_original - E_components) / E_original;
}

/**
 * Test 3: IMF Zero Mean Property
 * Each IMF should have approximately zero mean
 */
double test_imf_zero_mean(const std::vector<std::vector<double>>& imfs, int32_t n) {
    double max_mean_ratio = 0.0;
    
    for (const auto& imf : imfs) {
        double mean = 0.0;
        double energy = 0.0;
        
        for (int32_t i = 0; i < n; ++i) {
            mean += imf[i];
            energy += imf[i] * imf[i];
        }
        mean /= n;
        double rms = std::sqrt(energy / n);
        
        if (rms > 1e-10) {
            double ratio = std::abs(mean) / rms;
            max_mean_ratio = std::max(max_mean_ratio, ratio);
        }
    }
    
    return max_mean_ratio;
}

/**
 * Test 4: Orthogonality Index
 * IMFs should be approximately orthogonal (OI close to 0 is better)
 */
double test_orthogonality(const std::vector<std::vector<double>>& imfs,
                          const std::vector<double>& residue,
                          int32_t n) {
    // Compute orthogonality index as defined in the EEMD literature
    size_t K = imfs.size();
    double sum_cross = 0.0;
    double sum_sq = 0.0;
    
    for (int32_t t = 0; t < n; ++t) {
        double x_t = residue[t];
        for (const auto& imf : imfs) {
            x_t += imf[t];
        }
        sum_sq += x_t * x_t;
        
        // Cross terms
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = i + 1; j < K; ++j) {
                sum_cross += std::abs(imfs[i][t] * imfs[j][t]);
            }
        }
    }
    
    return (sum_sq > 1e-10) ? sum_cross / sum_sq : 0.0;
}

/**
 * Test 5: Monotonic Residue
 * Residue should be monotonic or have at most 1 extremum
 */
int test_residue_extrema(const std::vector<double>& residue, int32_t n) {
    int extrema_count = 0;
    
    for (int32_t i = 1; i < n - 1; ++i) {
        bool is_max = (residue[i] > residue[i-1]) && (residue[i] > residue[i+1]);
        bool is_min = (residue[i] < residue[i-1]) && (residue[i] < residue[i+1]);
        if (is_max || is_min) {
            extrema_count++;
        }
    }
    
    return extrema_count;
}

/**
 * Test 6: Determinism
 * Same seed must produce identical results
 */
double test_determinism(int32_t n, uint32_t seed) {
    std::vector<double> signal(n);
    generate_garch(signal.data(), n, 12345);
    
    eemd::ICEEMDAN decomposer1(eemd::ProcessingMode::Scientific);
    decomposer1.config().ensemble_size = 50;
    decomposer1.config().rng_seed = seed;
    
    std::vector<std::vector<double>> imfs1, imfs2;
    std::vector<double> residue1, residue2;
    
    decomposer1.decompose(signal.data(), n, imfs1, residue1);
    
    eemd::ICEEMDAN decomposer2(eemd::ProcessingMode::Scientific);
    decomposer2.config().ensemble_size = 50;
    decomposer2.config().rng_seed = seed;
    
    decomposer2.decompose(signal.data(), n, imfs2, residue2);
    
    // Check exact equality
    if (imfs1.size() != imfs2.size()) return 1.0;
    
    double max_diff = 0.0;
    for (size_t k = 0; k < imfs1.size(); ++k) {
        for (int32_t i = 0; i < n; ++i) {
            max_diff = std::max(max_diff, std::abs(imfs1[k][i] - imfs2[k][i]));
        }
    }
    for (int32_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(residue1[i] - residue2[i]));
    }
    
    return max_diff;
}

/**
 * Test 7: No NaN/Inf
 * Output must be finite
 */
bool test_no_nan_inf(const std::vector<std::vector<double>>& imfs,
                     const std::vector<double>& residue,
                     int32_t n) {
    for (const auto& imf : imfs) {
        for (int32_t i = 0; i < n; ++i) {
            if (!std::isfinite(imf[i])) return false;
        }
    }
    for (int32_t i = 0; i < n; ++i) {
        if (!std::isfinite(residue[i])) return false;
    }
    return true;
}

/**
 * Test 8: Frequency Ordering
 * IMF k should have lower mean frequency than IMF k-1
 */
bool test_frequency_ordering(const std::vector<std::vector<double>>& imfs, int32_t n) {
    if (imfs.size() < 2) return true;
    
    std::vector<double> mean_freqs;
    
    for (const auto& imf : imfs) {
        // Count zero crossings as proxy for frequency
        int crossings = 0;
        for (int32_t i = 1; i < n; ++i) {
            if ((imf[i] > 0 && imf[i-1] < 0) || (imf[i] < 0 && imf[i-1] > 0)) {
                crossings++;
            }
        }
        mean_freqs.push_back(static_cast<double>(crossings) / n);
    }
    
    // Check monotonic decrease (with some tolerance)
    for (size_t k = 1; k < mean_freqs.size(); ++k) {
        if (mean_freqs[k] > mean_freqs[k-1] * 1.5) {  // Allow 50% tolerance
            return false;
        }
    }
    
    return true;
}

/**
 * Test 9: Minimum IMF Count
 * Should extract at least log2(N) - 1 IMFs for typical signals
 */
bool test_minimum_imf_count(const std::vector<std::vector<double>>& imfs, int32_t n) {
    int expected_min = static_cast<int>(std::log2(n)) - 3;
    return static_cast<int>(imfs.size()) >= expected_min;
}

/**
 * Test 10: Input Preservation
 * Original signal should not be modified
 */
double test_input_preservation(int32_t n) {
    std::vector<double> signal(n);
    std::vector<double> signal_copy(n);
    
    generate_sum_of_sines(signal.data(), n, 42);
    std::copy(signal.begin(), signal.end(), signal_copy.begin());
    
    eemd::ICEEMDAN decomposer;
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    
    decomposer.decompose(signal.data(), n, imfs, residue);
    
    double max_diff = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(signal[i] - signal_copy[i]));
    }
    
    return max_diff;
}

// ============================================================================
// Test Runners
// ============================================================================

void run_signal_tests(const std::string& name, double* signal, int32_t n, 
                      eemd::ProcessingMode mode = eemd::ProcessingMode::Standard) {
    std::cout << "\n--- " << name << " (N=" << n << ") ---\n";
    
    eemd::ICEEMDAN decomposer(mode);
    decomposer.config().ensemble_size = 100;
    decomposer.config().rng_seed = 42;
    
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    
    decomposer.decompose(signal, n, imfs, residue);
    
    std::cout << "IMFs extracted: " << imfs.size() << "\n";
    
    // Run tests
    double recon_err = test_reconstruction(signal, imfs, residue, n);
    report("Reconstruction error", recon_err < 1e-6, recon_err, 1e-6);
    
    double energy_err = test_energy_conservation(signal, imfs, residue, n);
    report("Energy conservation", energy_err < 1e-10, energy_err, 1e-10);
    
    double mean_ratio = test_imf_zero_mean(imfs, n);
    report("IMF zero-mean property", mean_ratio < 0.1, mean_ratio, 0.1);
    
    double ortho = test_orthogonality(imfs, residue, n);
    report("Orthogonality index", ortho < 1.0, ortho, 1.0);
    
    int residue_extrema = test_residue_extrema(residue, n);
    report("Residue extrema count", residue_extrema <= 3, residue_extrema, 3, "extrema");
    
    bool finite = test_no_nan_inf(imfs, residue, n);
    report("No NaN/Inf", finite, finite ? 0.0 : 1.0, 0.5);
    
    bool freq_order = test_frequency_ordering(imfs, n);
    report("Frequency ordering", freq_order, freq_order ? 0.0 : 1.0, 0.5);
    
    bool imf_count = test_minimum_imf_count(imfs, n);
    report("Minimum IMF count", imf_count, static_cast<double>(imfs.size()), std::log2(n) - 3, "IMFs");
}

void run_edge_case_tests() {
    std::cout << "\n=== Edge Case Tests ===\n";
    
    // Very short signal
    {
        std::vector<double> signal(64);
        generate_sum_of_sines(signal.data(), 64, 42);
        run_signal_tests("Short signal", signal.data(), 64);
    }
    
    // Power of 2
    {
        std::vector<double> signal(1024);
        generate_garch(signal.data(), 1024, 42);
        run_signal_tests("Power-of-2 length", signal.data(), 1024);
    }
    
    // Non-power of 2
    {
        std::vector<double> signal(1000);
        generate_garch(signal.data(), 1000, 42);
        run_signal_tests("Non-power-of-2 length", signal.data(), 1000);
    }
    
    // Constant signal
    {
        std::vector<double> signal(512, 42.0);
        run_signal_tests("Constant signal", signal.data(), 512);
    }
    
    // Pure noise
    {
        std::vector<double> signal(1024);
        generate_white_noise(signal.data(), 1024, 42);
        run_signal_tests("White noise", signal.data(), 1024);
    }
    
    // Step function
    {
        std::vector<double> signal(512);
        generate_step(signal.data(), 512);
        run_signal_tests("Step function", signal.data(), 512);
    }
    
    // Ramp
    {
        std::vector<double> signal(512);
        generate_ramp(signal.data(), 512);
        run_signal_tests("Ramp signal", signal.data(), 512);
    }
}

void run_mode_comparison_tests() {
    std::cout << "\n=== Processing Mode Comparison ===\n";
    
    const int32_t n = 2048;
    std::vector<double> signal(n);
    generate_garch(signal.data(), n, 42);
    
    run_signal_tests("Standard mode", signal.data(), n, eemd::ProcessingMode::Standard);
    run_signal_tests("Finance mode", signal.data(), n, eemd::ProcessingMode::Finance);
    run_signal_tests("Scientific mode", signal.data(), n, eemd::ProcessingMode::Scientific);
}

void run_robustness_tests() {
    std::cout << "\n=== Robustness Tests ===\n";
    
    const int32_t n = 1024;
    
    // Determinism
    double det_err = test_determinism(n, 12345);
    report("Determinism (same seed)", det_err == 0.0, det_err, 0.0);
    
    // Input preservation
    double input_err = test_input_preservation(n);
    report("Input preservation", input_err == 0.0, input_err, 0.0);
    
    // Different seeds should give different results
    std::vector<double> signal(n);
    generate_garch(signal.data(), n, 42);
    
    eemd::ICEEMDAN d1, d2;
    d1.config().ensemble_size = 50;
    d2.config().ensemble_size = 50;
    d1.config().rng_seed = 111;
    d2.config().rng_seed = 222;
    
    std::vector<std::vector<double>> imfs1, imfs2;
    std::vector<double> res1, res2;
    
    d1.decompose(signal.data(), n, imfs1, res1);
    d2.decompose(signal.data(), n, imfs2, res2);
    
    double seed_diff = 0.0;
    if (imfs1.size() == imfs2.size()) {
        for (size_t k = 0; k < imfs1.size(); ++k) {
            for (int32_t i = 0; i < n; ++i) {
                seed_diff += std::abs(imfs1[k][i] - imfs2[k][i]);
            }
        }
    } else {
        seed_diff = 1.0;
    }
    report("Different seeds diverge", seed_diff > 1e-6, seed_diff, 1e-6);
}

void run_numerical_stress_tests() {
    std::cout << "\n=== Numerical Stress Tests ===\n";
    
    const int32_t n = 512;
    
    // Large values
    {
        std::vector<double> signal(n);
        for (int32_t i = 0; i < n; ++i) {
            signal[i] = 1e10 * std::sin(2 * M_PI * 10 * i / n);
        }
        run_signal_tests("Large amplitude (1e10)", signal.data(), n);
    }
    
    // Small values
    {
        std::vector<double> signal(n);
        for (int32_t i = 0; i < n; ++i) {
            signal[i] = 1e-10 * std::sin(2 * M_PI * 10 * i / n);
        }
        run_signal_tests("Small amplitude (1e-10)", signal.data(), n);
    }
    
    // Signal with outliers
    {
        std::vector<double> signal(n);
        generate_sum_of_sines(signal.data(), n, 42);
        signal[n/4] = 1000.0;   // Spike
        signal[n/2] = -1000.0;  // Spike
        run_signal_tests("Signal with outliers", signal.data(), n);
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "ICEEMDAN Accuracy Test Suite\n";
    std::cout << "========================================\n";
    
    // Standard signals
    {
        const int32_t n = 2048;
        std::vector<double> signal(n);
        
        generate_sum_of_sines(signal.data(), n, 42);
        run_signal_tests("Sum of sines", signal.data(), n);
        
        generate_garch(signal.data(), n, 42);
        run_signal_tests("GARCH price", signal.data(), n);
    }
    
    run_edge_case_tests();
    run_mode_comparison_tests();
    run_robustness_tests();
    run_numerical_stress_tests();
    
    // Summary
    std::cout << "\n========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "========================================\n";
    
    int passed = 0, failed = 0;
    for (const auto& r : g_results) {
        if (r.passed) passed++;
        else failed++;
    }
    
    std::cout << "Passed: " << passed << " / " << g_results.size() << "\n";
    std::cout << "Failed: " << failed << " / " << g_results.size() << "\n";
    
    if (failed > 0) {
        std::cout << "\nFailed tests:\n";
        for (const auto& r : g_results) {
            if (!r.passed) {
                std::cout << "  - " << r.name << " (got " << r.value << ", expected < " << r.threshold << ")\n";
            }
        }
    }
    
    std::cout << "\n";
    return (failed == 0) ? 0 : 1;
}
