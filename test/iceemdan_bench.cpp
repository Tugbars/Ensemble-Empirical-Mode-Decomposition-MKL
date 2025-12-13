/**
 * ICEEMDAN-MKL Performance Benchmark
 *
 * Tests scaling across signal lengths, ensemble sizes, thread counts,
 * and compares EEMD vs ICEEMDAN performance and quality.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "iceemdan_mkl.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>

using namespace eemd;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Sample Entropy (for IMF analysis)
// ============================================================================

inline double compute_sample_entropy(const double *signal, int32_t n, int32_t m = 2, double r_factor = 0.2)
{
    if (n < m + 2) return 0.0;
    
    // Compute standard deviation for tolerance
    double mean = 0.0;
    for (int32_t i = 0; i < n; ++i) mean += signal[i];
    mean /= n;
    
    double var = 0.0;
    for (int32_t i = 0; i < n; ++i)
    {
        double d = signal[i] - mean;
        var += d * d;
    }
    double sd = std::sqrt(var / n);
    double r = r_factor * sd;
    
    if (r < 1e-10) return 0.0;
    
    // Count template matches
    auto count_matches = [&](int32_t template_len) -> int64_t {
        int64_t count = 0;
        for (int32_t i = 0; i < n - template_len; ++i)
        {
            for (int32_t j = i + 1; j < n - template_len; ++j)
            {
                bool match = true;
                for (int32_t k = 0; k < template_len && match; ++k)
                {
                    if (std::abs(signal[i + k] - signal[j + k]) > r)
                        match = false;
                }
                if (match) ++count;
            }
        }
        return count;
    };
    
    int64_t A = count_matches(m + 1);
    int64_t B = count_matches(m);
    
    if (B == 0 || A == 0) return 0.0;
    
    return -std::log(static_cast<double>(A) / B);
}

// ============================================================================
// Test Signal Generators
// ============================================================================

// Multi-component synthetic signal (known structure)
void generate_synthetic_signal(double *signal, int32_t n, double dt)
{
    for (int32_t i = 0; i < n; ++i)
    {
        const double t = i * dt;
        // Trend (slow)
        const double trend = 2.0 * std::sin(2.0 * M_PI * 0.5 * t);
        // Mid-frequency with AM
        const double am = 1.0 + 0.3 * std::sin(2.0 * M_PI * 1.0 * t);
        const double mid = am * std::sin(2.0 * M_PI * 5.0 * t);
        // High frequency
        const double high = 0.5 * std::sin(2.0 * M_PI * 25.0 * t);
        // Chirp (frequency sweep)
        const double chirp = 0.3 * std::sin(2.0 * M_PI * (10.0 + 5.0 * t) * t);
        
        signal[i] = trend + mid + high + chirp;
    }
}

// Simulated financial returns (regime switching + noise)
void generate_financial_signal(double *signal, int32_t n, uint32_t seed)
{
    VSLStreamStatePtr stream = nullptr;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    
    // Generate base noise
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, signal, 0.0, 1.0);
    
    // Add regime-dependent volatility
    double vol = 0.01;
    for (int32_t i = 1; i < n; ++i)
    {
        // Regime switch probability
        double regime_prob = 0.02;
        double u;
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &u, 0.0, 1.0);
        
        if (u < regime_prob)
        {
            vol = (vol < 0.02) ? 0.04 : 0.01; // Switch regime
        }
        
        signal[i] = signal[i] * vol;
        
        // Add slow trend
        signal[i] += 0.0001 * std::sin(2.0 * M_PI * i / n * 3);
    }
    
    // Cumsum to get price-like series
    for (int32_t i = 1; i < n; ++i)
    {
        signal[i] += signal[i - 1];
    }
    
    vslDeleteStream(&stream);
}

// ============================================================================
// Benchmark Structures
// ============================================================================

struct BenchResult
{
    double time_ms;
    double throughput_msamples;
    int32_t n_imfs;
    double noise_bank_time_ms; // ICEEMDAN-specific
};

struct QualityMetrics
{
    double orthogonality_index;  // How orthogonal are IMFs (0 = perfect)
    double reconstruction_error; // ||signal - sum(IMFs) - residue||
    double mode_mixing_score;    // Heuristic for mode mixing
};

// ============================================================================
// Quality Metrics
// ============================================================================

QualityMetrics compute_quality(
    const double *signal,
    int32_t n,
    const std::vector<std::vector<double>> &imfs,
    const std::vector<double> &residue)
{
    QualityMetrics q;
    
    // Orthogonality index: sum of |<IMF_i, IMF_j>| / (||IMF_i|| * ||IMF_j||)
    double orth_sum = 0.0;
    int32_t orth_count = 0;
    
    for (size_t i = 0; i < imfs.size(); ++i)
    {
        for (size_t j = i + 1; j < imfs.size(); ++j)
        {
            double dot = 0.0, norm_i = 0.0, norm_j = 0.0;
            
            for (int32_t k = 0; k < n; ++k)
            {
                dot += imfs[i][k] * imfs[j][k];
                norm_i += imfs[i][k] * imfs[i][k];
                norm_j += imfs[j][k] * imfs[j][k];
            }
            
            if (norm_i > 1e-10 && norm_j > 1e-10)
            {
                orth_sum += std::abs(dot) / std::sqrt(norm_i * norm_j);
                ++orth_count;
            }
        }
    }
    
    q.orthogonality_index = (orth_count > 0) ? orth_sum / orth_count : 0.0;
    
    // Reconstruction error
    std::vector<double> reconstructed(n, 0.0);
    for (const auto &imf : imfs)
    {
        for (int32_t i = 0; i < n; ++i)
        {
            reconstructed[i] += imf[i];
        }
    }
    for (int32_t i = 0; i < n; ++i)
    {
        reconstructed[i] += residue[i];
    }
    
    double err_sum = 0.0, sig_sum = 0.0;
    for (int32_t i = 0; i < n; ++i)
    {
        double diff = signal[i] - reconstructed[i];
        err_sum += diff * diff;
        sig_sum += signal[i] * signal[i];
    }
    
    q.reconstruction_error = (sig_sum > 1e-10) ? std::sqrt(err_sum / sig_sum) : 0.0;
    
    // Mode mixing score: variance of instantaneous frequency within each IMF
    // Lower variance = less mode mixing
    double mm_score = 0.0;
    int32_t mm_count = 0;
    
    for (const auto &imf : imfs)
    {
        std::vector<double> inst_freq(n);
        if (compute_instantaneous_frequency(imf.data(), n, inst_freq.data(), 1.0))
        {
            // Compute variance of instantaneous frequency
            double mean = 0.0;
            for (int32_t i = 0; i < n; ++i)
            {
                mean += std::abs(inst_freq[i]);
            }
            mean /= n;
            
            double var = 0.0;
            for (int32_t i = 0; i < n; ++i)
            {
                double d = std::abs(inst_freq[i]) - mean;
                var += d * d;
            }
            var /= n;
            
            // Normalized by mean (coefficient of variation)
            if (mean > 1e-10)
            {
                mm_score += std::sqrt(var) / mean;
                ++mm_count;
            }
        }
    }
    
    q.mode_mixing_score = (mm_count > 0) ? mm_score / mm_count : 0.0;
    
    return q;
}

// ============================================================================
// ICEEMDAN Benchmarks
// ============================================================================

BenchResult run_iceemdan_benchmark(int32_t n, int32_t ensemble_size, int32_t n_trials)
{
    std::vector<double> signal(n);
    generate_synthetic_signal(signal.data(), n, 0.01);
    
    ICEEMDANConfig config;
    config.max_imfs = 8;
    config.ensemble_size = ensemble_size;
    config.noise_std = 0.2;
    
    ICEEMDAN decomposer(config);
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    
    // Warmup (includes noise bank initialization)
    auto t_nb_start = Clock::now();
    decomposer.decompose(signal.data(), n, imfs, residue);
    auto t_nb_end = Clock::now();
    double noise_bank_time = std::chrono::duration<double, std::milli>(t_nb_end - t_nb_start).count();
    
    // Timed runs
    double total_time = 0.0;
    int32_t n_imfs = 0;
    
    for (int32_t t = 0; t < n_trials; ++t)
    {
        auto t0 = Clock::now();
        decomposer.decompose(signal.data(), n, imfs, residue);
        auto t1 = Clock::now();
        total_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
        n_imfs = static_cast<int32_t>(imfs.size());
    }
    
    const double avg_time = total_time / n_trials;
    const double total_samples = static_cast<double>(ensemble_size) * n;
    const double throughput = (total_samples / 1e6) / (avg_time / 1000.0);
    
    return {avg_time, throughput, n_imfs, noise_bank_time};
}

void bench_signal_length()
{
    std::cout << "\n=== ICEEMDAN: Signal Length Scaling ===\n";
    std::cout << "Ensemble: 100, Threads: " << omp_get_max_threads() << "\n\n";
    
    std::cout << std::setw(10) << "Length"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Throughput"
              << std::setw(8) << "IMFs"
              << std::setw(14) << "NoiseBnk(ms)" << "\n";
    std::cout << std::string(59, '-') << "\n";
    
    for (int32_t n : {256, 512, 1024, 2048, 4096, 8192})
    {
        auto r = run_iceemdan_benchmark(n, 100, 3);
        std::cout << std::setw(10) << n
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.time_ms
                  << std::setw(12) << std::setprecision(2) << r.throughput_msamples << " MS/s"
                  << std::setw(8) << r.n_imfs
                  << std::setw(14) << std::setprecision(1) << r.noise_bank_time_ms << "\n";
    }
}

void bench_ensemble_size()
{
    std::cout << "\n=== ICEEMDAN: Ensemble Size Scaling ===\n";
    std::cout << "Signal: 1024, Threads: " << omp_get_max_threads() << "\n\n";
    
    std::cout << std::setw(10) << "Ensemble"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "ms/trial" << "\n";
    std::cout << std::string(49, '-') << "\n";
    
    for (int32_t ens : {10, 25, 50, 100, 200, 500})
    {
        auto r = run_iceemdan_benchmark(1024, ens, 3);
        std::cout << std::setw(10) << ens
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.time_ms
                  << std::setw(12) << std::setprecision(2) << r.throughput_msamples << " MS/s"
                  << std::setw(12) << std::setprecision(3) << r.time_ms / ens << "\n";
    }
}

void bench_thread_scaling()
{
    std::cout << "\n=== ICEEMDAN: Thread Scaling ===\n";
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
        mkl_set_num_threads(1);
        
        // Warm up thread pool
        #pragma omp parallel
        {
            volatile int x = 0;
            (void)x;
        }
        
        auto r = run_iceemdan_benchmark(1024, 100, 5);
        
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
    
    omp_set_num_threads(max_threads);
    mkl_set_num_threads(1);
}

void bench_eemd_vs_iceemdan()
{
    std::cout << "\n=== EEMD vs ICEEMDAN Comparison ===\n";
    std::cout << "Signal: 2048, Ensemble: 100\n\n";
    
    const int32_t n = 2048;
    std::vector<double> signal(n);
    generate_synthetic_signal(signal.data(), n, 0.01);
    
    // EEMD
    EEMDConfig eemd_config;
    eemd_config.max_imfs = 8;
    eemd_config.ensemble_size = 100;
    eemd_config.noise_std = 0.2;
    
    EEMD eemd_decomposer(eemd_config);
    std::vector<std::vector<double>> eemd_imfs;
    int32_t eemd_n_imfs;
    
    // Warmup
    eemd_decomposer.decompose(signal.data(), n, eemd_imfs, eemd_n_imfs);
    
    auto t0 = Clock::now();
    for (int i = 0; i < 5; ++i)
    {
        eemd_decomposer.decompose(signal.data(), n, eemd_imfs, eemd_n_imfs);
    }
    auto t1 = Clock::now();
    double eemd_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
    
    // Build residue for EEMD (sum of what's left)
    std::vector<double> eemd_residue(n, 0.0);
    for (int32_t i = 0; i < n; ++i)
    {
        eemd_residue[i] = signal[i];
        for (const auto &imf : eemd_imfs)
        {
            eemd_residue[i] -= imf[i];
        }
    }
    
    // ICEEMDAN
    ICEEMDANConfig iceemdan_config;
    iceemdan_config.max_imfs = 8;
    iceemdan_config.ensemble_size = 100;
    iceemdan_config.noise_std = 0.2;
    
    ICEEMDAN iceemdan_decomposer(iceemdan_config);
    std::vector<std::vector<double>> iceemdan_imfs;
    std::vector<double> iceemdan_residue;
    
    // Warmup
    iceemdan_decomposer.decompose(signal.data(), n, iceemdan_imfs, iceemdan_residue);
    
    t0 = Clock::now();
    for (int i = 0; i < 5; ++i)
    {
        iceemdan_decomposer.decompose(signal.data(), n, iceemdan_imfs, iceemdan_residue);
    }
    t1 = Clock::now();
    double iceemdan_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
    
    // Quality metrics
    auto eemd_quality = compute_quality(signal.data(), n, eemd_imfs, eemd_residue);
    auto iceemdan_quality = compute_quality(signal.data(), n, iceemdan_imfs, iceemdan_residue);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "                    EEMD        ICEEMDAN\n";
    std::cout << "                    ----        --------\n";
    std::cout << "Time (ms):          " << std::setw(8) << eemd_time
              << "    " << std::setw(8) << iceemdan_time << "\n";
    std::cout << "IMFs extracted:     " << std::setw(8) << eemd_n_imfs
              << "    " << std::setw(8) << iceemdan_imfs.size() << "\n";
    std::cout << std::setprecision(6);
    std::cout << "Recon error:        " << std::setw(8) << eemd_quality.reconstruction_error
              << "    " << std::setw(8) << iceemdan_quality.reconstruction_error << "\n";
    std::cout << std::setprecision(4);
    std::cout << "Orthogonality:      " << std::setw(8) << eemd_quality.orthogonality_index
              << "    " << std::setw(8) << iceemdan_quality.orthogonality_index << "\n";
    std::cout << "Mode mixing:        " << std::setw(8) << eemd_quality.mode_mixing_score
              << "    " << std::setw(8) << iceemdan_quality.mode_mixing_score << "\n";
    
    std::cout << "\n(Lower orthogonality & mode mixing = better)\n";
}

void bench_analysis_utilities()
{
    std::cout << "\n=== Analysis Utilities Benchmark ===\n";
    std::cout << "Signal: 2048\n\n";
    
    const int32_t n = 2048;
    std::vector<double> signal(n);
    generate_synthetic_signal(signal.data(), n, 0.01);
    
    // Decompose first
    ICEEMDANConfig config;
    config.max_imfs = 8;
    config.ensemble_size = 50;
    
    ICEEMDAN decomposer(config);
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    decomposer.decompose(signal.data(), n, imfs, residue);
    
    std::cout << "Extracted " << imfs.size() << " IMFs\n\n";
    
    // Benchmark analysis functions
    const int32_t n_trials = 10;
    
    // Hurst exponent
    auto t0 = Clock::now();
    for (int t = 0; t < n_trials; ++t)
    {
        for (const auto &imf : imfs)
        {
            volatile double h = estimate_hurst_rs(imf.data(), n);
            (void)h;
        }
    }
    auto t1 = Clock::now();
    double hurst_time = std::chrono::duration<double, std::micro>(t1 - t0).count() / n_trials / imfs.size();
    
    // Spectral entropy
    t0 = Clock::now();
    for (int t = 0; t < n_trials; ++t)
    {
        for (const auto &imf : imfs)
        {
            volatile double se = compute_spectral_entropy(imf.data(), n);
            (void)se;
        }
    }
    t1 = Clock::now();
    double spectral_time = std::chrono::duration<double, std::micro>(t1 - t0).count() / n_trials / imfs.size();
    
    // Sample entropy (expensive!)
    t0 = Clock::now();
    for (const auto &imf : imfs)
    {
        volatile double se = compute_sample_entropy(imf.data(), n);
        (void)se;
    }
    t1 = Clock::now();
    double sample_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / imfs.size();
    
    // Full IMF analysis
    t0 = Clock::now();
    for (int t = 0; t < n_trials; ++t)
    {
        for (const auto &imf : imfs)
        {
            volatile auto analysis = analyze_imf(imf.data(), n);
            (void)analysis;
        }
    }
    t1 = Clock::now();
    double full_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_trials / imfs.size();
    
    std::cout << std::fixed;
    std::cout << "Hurst (R/S):        " << std::setprecision(1) << std::setw(8) << hurst_time << " μs/IMF\n";
    std::cout << "Spectral entropy:   " << std::setw(8) << spectral_time << " μs/IMF\n";
    std::cout << "Sample entropy:     " << std::setprecision(1) << std::setw(8) << sample_time << " ms/IMF  (O(N²) warning!)\n";
    std::cout << "Full analyze_imf:   " << std::setprecision(2) << std::setw(8) << full_time << " ms/IMF\n";
    
    // Print actual analysis for first few IMFs
    std::cout << "\n--- IMF Analysis Results ---\n";
    std::cout << std::setw(6) << "IMF"
              << std::setw(10) << "Hurst"
              << std::setw(12) << "Spec.Ent"
              << std::setw(12) << "Energy"
              << std::setw(12) << "MeanFreq"
              << std::setw(10) << "Type" << "\n";
    std::cout << std::string(62, '-') << "\n";
    
    for (size_t i = 0; i < imfs.size(); ++i)
    {
        auto a = analyze_imf(imfs[i].data(), n, 100.0); // 100 Hz sample rate
        
        std::cout << std::setw(6) << i
                  << std::setw(10) << std::setprecision(3) << a.hurst
                  << std::setw(12) << std::setprecision(3) << a.spectral_entropy
                  << std::setw(12) << std::setprecision(2) << a.energy
                  << std::setw(12) << std::setprecision(2) << a.mean_frequency
                  << std::setw(10) << (a.likely_noise ? "NOISE" : "SIGNAL") << "\n";
    }
}

// ============================================================================
// Detailed Cycle Breakdown
// ============================================================================

/**
 * Instrumented ICEEMDAN for profiling where cycles go
 * 
 * Breakdown:
 * 1. Noise bank initialization (one-time)
 *    - Noise generation
 *    - EMD of each noise realization
 * 2. Per-IMF stage:
 *    - Perturbation (signal + noise_imf)
 *    - Local mean computation (spline construct + evaluate)
 *    - Reduction (thread accumulation + averaging)
 * 3. Overhead:
 *    - Memory copies
 *    - Synchronization barriers
 */

struct CycleBreakdown
{
    // Noise bank (one-time)
    double noise_gen_ms = 0.0;
    double noise_emd_ms = 0.0;
    double noise_bank_total_ms = 0.0;
    
    // Per-IMF averages
    double perturbation_ms = 0.0;
    double spline_construct_ms = 0.0;
    double spline_evaluate_ms = 0.0;
    double reduction_ms = 0.0;
    double imf_extract_ms = 0.0;
    
    // Totals
    double decompose_total_ms = 0.0;
    int32_t n_imfs = 0;
    int32_t n_stages = 0;
    
    void print() const
    {
        std::cout << std::fixed << std::setprecision(2);
        
        std::cout << "\n--- Noise Bank (One-Time) ---\n";
        std::cout << "  Noise generation:     " << std::setw(8) << noise_gen_ms << " ms\n";
        std::cout << "  Noise EMD:            " << std::setw(8) << noise_emd_ms << " ms\n";
        std::cout << "  Total:                " << std::setw(8) << noise_bank_total_ms << " ms\n";
        
        std::cout << "\n--- Per-IMF Stage (avg of " << n_stages << " stages) ---\n";
        std::cout << "  Perturbation:         " << std::setw(8) << perturbation_ms << " ms\n";
        std::cout << "  Spline construct:     " << std::setw(8) << spline_construct_ms << " ms\n";
        std::cout << "  Spline evaluate:      " << std::setw(8) << spline_evaluate_ms << " ms\n";
        std::cout << "  Reduction:            " << std::setw(8) << reduction_ms << " ms\n";
        std::cout << "  IMF extraction:       " << std::setw(8) << imf_extract_ms << " ms\n";
        
        double per_imf_total = perturbation_ms + spline_construct_ms + spline_evaluate_ms + 
                               reduction_ms + imf_extract_ms;
        std::cout << "  Per-IMF total:        " << std::setw(8) << per_imf_total << " ms\n";
        
        std::cout << "\n--- Summary ---\n";
        std::cout << "  Decomposition total:  " << std::setw(8) << decompose_total_ms << " ms\n";
        std::cout << "  IMFs extracted:       " << std::setw(8) << n_imfs << "\n";
        
        // Percentage breakdown
        double total = noise_bank_total_ms + decompose_total_ms;
        std::cout << "\n--- Time Distribution ---\n";
        std::cout << "  Noise bank:           " << std::setw(7) << std::setprecision(1) 
                  << 100.0 * noise_bank_total_ms / total << " %\n";
        std::cout << "  Decomposition:        " << std::setw(7) 
                  << 100.0 * decompose_total_ms / total << " %\n";
        
        if (n_stages > 0)
        {
            double spline_total = (spline_construct_ms + spline_evaluate_ms) * n_stages;
            double compute_total = perturbation_ms * n_stages + spline_total + 
                                   reduction_ms * n_stages;
            std::cout << "\n--- Decomposition Breakdown ---\n";
            std::cout << "  Splines (construct):  " << std::setw(7) 
                      << 100.0 * spline_construct_ms * n_stages / decompose_total_ms << " %\n";
            std::cout << "  Splines (evaluate):   " << std::setw(7) 
                      << 100.0 * spline_evaluate_ms * n_stages / decompose_total_ms << " %\n";
            std::cout << "  Perturbation:         " << std::setw(7) 
                      << 100.0 * perturbation_ms * n_stages / decompose_total_ms << " %\n";
            std::cout << "  Reduction:            " << std::setw(7) 
                      << 100.0 * reduction_ms * n_stages / decompose_total_ms << " %\n";
        }
    }
};

/**
 * Instrumented Local Mean Computer with timing
 */
class LocalMeanComputerProfiled
{
public:
    explicit LocalMeanComputerProfiled(int32_t max_len, int32_t boundary_extend)
        : max_len_(max_len), boundary_extend_(boundary_extend),
          max_idx_(max_len / 2 + 2), min_idx_(max_len / 2 + 2),
          ext_x_(max_len + 20), ext_y_(max_len + 20),
          upper_env_(max_len), lower_env_(max_len)
    {
    }

    bool compute(const double *signal, int32_t n, double *local_mean,
                 double &construct_time_us, double &evaluate_time_us)
    {
        construct_time_us = 0.0;
        evaluate_time_us = 0.0;
        
        // Find extrema
        find_maxima_raw(signal, n, max_idx_.data(), n_max_);
        find_minima_raw(signal, n, min_idx_.data(), n_min_);

        if (n_max_ < 2 || n_min_ < 2)
        {
            std::memcpy(local_mean, signal, n * sizeof(double));
            return false;
        }

        int32_t n_ext, ext_start;
        
        // Upper envelope - timed
        extend_extrema_raw(max_idx_.data(), n_max_, signal, n,
                           boundary_extend_, ext_x_.data(), ext_y_.data(),
                           n_ext, ext_start);

        auto t0 = Clock::now();
        if (!upper_spline_.construct(ext_x_.data(), ext_y_.data(), n_ext))
        {
            std::memcpy(local_mean, signal, n * sizeof(double));
            return false;
        }
        auto t1 = Clock::now();
        construct_time_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        
        t0 = Clock::now();
        if (!upper_spline_.evaluate_uniform(upper_env_.data, n))
        {
            std::memcpy(local_mean, signal, n * sizeof(double));
            return false;
        }
        t1 = Clock::now();
        evaluate_time_us += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // Lower envelope - timed
        extend_extrema_raw(min_idx_.data(), n_min_, signal, n,
                           boundary_extend_, ext_x_.data(), ext_y_.data(),
                           n_ext, ext_start);

        t0 = Clock::now();
        if (!lower_spline_.construct(ext_x_.data(), ext_y_.data(), n_ext))
        {
            std::memcpy(local_mean, signal, n * sizeof(double));
            return false;
        }
        t1 = Clock::now();
        construct_time_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        
        t0 = Clock::now();
        if (!lower_spline_.evaluate_uniform(lower_env_.data, n))
        {
            std::memcpy(local_mean, signal, n * sizeof(double));
            return false;
        }
        t1 = Clock::now();
        evaluate_time_us += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // Local mean
        const double *__restrict upper = upper_env_.data;
        const double *__restrict lower = lower_env_.data;
        double *__restrict out = local_mean;

        EEMD_OMP_SIMD
        for (int32_t i = 0; i < n; ++i)
        {
            out[i] = 0.5 * (upper[i] + lower[i]);
        }

        return true;
    }

private:
    int32_t max_len_;
    int32_t boundary_extend_;

    std::vector<int32_t> max_idx_;
    std::vector<int32_t> min_idx_;
    std::vector<double> ext_x_;
    std::vector<double> ext_y_;

    int32_t n_max_ = 0;
    int32_t n_min_ = 0;

    AlignedBuffer<double> upper_env_;
    AlignedBuffer<double> lower_env_;

    MKLSpline upper_spline_;
    MKLSpline lower_spline_;
};

/**
 * Profile noise bank initialization
 */
CycleBreakdown profile_noise_bank(int32_t n, int32_t ensemble_size, int32_t max_imfs,
                                   const EEMDConfig &emd_config, uint32_t seed)
{
    CycleBreakdown bd;
    
    // Allocate storage
    std::vector<std::vector<std::vector<double>>> noise_imfs(ensemble_size);
    std::vector<int32_t> imf_counts(ensemble_size);
    
    for (int32_t i = 0; i < ensemble_size; ++i)
    {
        noise_imfs[i].resize(max_imfs);
        for (int32_t k = 0; k < max_imfs; ++k)
        {
            noise_imfs[i][k].resize(n);
        }
    }
    
    // Profile noise generation separately
    auto t_gen_start = Clock::now();
    
    std::vector<AlignedBuffer<double>> all_noise(ensemble_size);
    for (int32_t i = 0; i < ensemble_size; ++i)
    {
        all_noise[i].resize(n);
    }
    
    #pragma omp parallel
    {
        const int32_t tid = omp_get_thread_num();
        VSLStreamStatePtr stream = nullptr;
        vslNewStream(&stream, VSL_BRNG_MT19937, seed + tid * 10000);
        
        #pragma omp for schedule(static)
        for (int32_t i = 0; i < ensemble_size; ++i)
        {
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream,
                          n, all_noise[i].data, 0.0, 1.0);
        }
        
        vslDeleteStream(&stream);
    }
    
    auto t_gen_end = Clock::now();
    bd.noise_gen_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
    
    // Profile EMD of noise
    auto t_emd_start = Clock::now();
    
    #pragma omp parallel
    {
        Sifter sifter(n, emd_config);
        AlignedBuffer<double> work(n);
        
        #pragma omp for schedule(dynamic)
        for (int32_t i = 0; i < ensemble_size; ++i)
        {
            std::memcpy(work.data, all_noise[i].data, n * sizeof(double));
            
            int32_t imf_count = 0;
            for (int32_t k = 0; k < max_imfs; ++k)
            {
                if (!sifter.sift_imf(work.data, noise_imfs[i][k].data(), n))
                {
                    break;
                }
                ++imf_count;
            }
            imf_counts[i] = imf_count;
        }
    }
    
    auto t_emd_end = Clock::now();
    bd.noise_emd_ms = std::chrono::duration<double, std::milli>(t_emd_end - t_emd_start).count();
    
    bd.noise_bank_total_ms = bd.noise_gen_ms + bd.noise_emd_ms;
    
    return bd;
}

/**
 * Profile a single ICEEMDAN decomposition with detailed timing
 */
CycleBreakdown profile_iceemdan_decomposition(
    const double *signal,
    int32_t n,
    const ICEEMDANConfig &config)
{
    CycleBreakdown bd;
    
    // Build EMD config
    EEMDConfig emd_config;
    emd_config.max_imfs = config.max_imfs;
    emd_config.max_sift_iters = config.max_sift_iters;
    emd_config.sift_threshold = config.sift_threshold;
    emd_config.boundary_extend = config.boundary_extend;
    
    // Profile noise bank
    auto t_nb_start = Clock::now();
    CycleBreakdown nb_profile = profile_noise_bank(n, config.ensemble_size, config.max_imfs,
                                                    emd_config, config.rng_seed);
    auto t_nb_end = Clock::now();
    
    bd.noise_gen_ms = nb_profile.noise_gen_ms;
    bd.noise_emd_ms = nb_profile.noise_emd_ms;
    bd.noise_bank_total_ms = nb_profile.noise_bank_total_ms;
    
    // Now profile the actual decomposition stages
    // We need to rebuild noise bank (simpler approach than passing it)
    StandardNoiseBank noise_bank;
    noise_bank.initialize(n, config.ensemble_size, config.max_imfs,
                          emd_config, config.rng_seed);
    
    const double signal_std = compute_std(signal, n);
    double noise_amplitude = config.noise_std * signal_std;
    
    AlignedBuffer<double> r_current(n);
    std::memcpy(r_current.data, signal, n * sizeof(double));
    
    // Accumulators for per-stage timing
    double total_perturbation_us = 0.0;
    double total_spline_construct_us = 0.0;
    double total_spline_evaluate_us = 0.0;
    double total_reduction_us = 0.0;
    double total_imf_extract_us = 0.0;
    
    std::vector<AlignedBuffer<double>> imf_storage(config.max_imfs);
    for (auto &buf : imf_storage)
    {
        buf.resize(n);
    }
    
    AlignedBuffer<double> mean_accumulator(n);
    int32_t actual_imf_count = 0;
    
    auto t_decompose_start = Clock::now();
    
    for (int32_t k = 0; k < config.max_imfs; ++k)
    {
        mean_accumulator.zero();
        int32_t valid_trials = 0;
        
        double stage_perturbation_us = 0.0;
        double stage_construct_us = 0.0;
        double stage_evaluate_us = 0.0;
        
        // Ensemble loop (sequential for accurate timing)
        for (int32_t i = 0; i < config.ensemble_size; ++i)
        {
            AlignedBuffer<double> perturbed(n);
            AlignedBuffer<double> local_mean(n);
            
            const double *noise_imf = noise_bank.get_noise_imf(i, k);
            
            // Time perturbation
            auto t0 = Clock::now();
            if (!noise_imf)
            {
                std::memcpy(perturbed.data, r_current.data, n * sizeof(double));
            }
            else
            {
                const double *__restrict r = r_current.data;
                const double *__restrict nz = noise_imf;
                double *__restrict p = perturbed.data;
                
                EEMD_OMP_SIMD
                for (int32_t j = 0; j < n; ++j)
                {
                    p[j] = r[j] + noise_amplitude * nz[j];
                }
            }
            auto t1 = Clock::now();
            stage_perturbation_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            
            // Time local mean computation (includes spline timing)
            double construct_us = 0.0, evaluate_us = 0.0;
            LocalMeanComputerProfiled lm_computer(n, config.boundary_extend);
            
            if (lm_computer.compute(perturbed.data, n, local_mean.data, construct_us, evaluate_us))
            {
                ++valid_trials;
                
                // Accumulate
                double *__restrict acc = mean_accumulator.data;
                const double *__restrict lm = local_mean.data;
                
                for (int32_t j = 0; j < n; ++j)
                {
                    acc[j] += lm[j];
                }
            }
            
            stage_construct_us += construct_us;
            stage_evaluate_us += evaluate_us;
        }
        
        if (valid_trials == 0)
            break;
        
        // Time reduction (averaging)
        auto t_red_start = Clock::now();
        const double scale = 1.0 / valid_trials;
        double *__restrict acc = mean_accumulator.data;
        
        EEMD_OMP_SIMD
        for (int32_t j = 0; j < n; ++j)
        {
            acc[j] *= scale;
        }
        auto t_red_end = Clock::now();
        double stage_reduction_us = std::chrono::duration<double, std::micro>(t_red_end - t_red_start).count();
        
        // Time IMF extraction
        auto t_imf_start = Clock::now();
        const double *__restrict r = r_current.data;
        const double *__restrict m = mean_accumulator.data;
        double *__restrict out = imf_storage[actual_imf_count].data;
        
        EEMD_OMP_SIMD
        for (int32_t j = 0; j < n; ++j)
        {
            out[j] = r[j] - m[j];
        }
        
        // Update residue
        std::memcpy(r_current.data, mean_accumulator.data, n * sizeof(double));
        auto t_imf_end = Clock::now();
        double stage_imf_extract_us = std::chrono::duration<double, std::micro>(t_imf_end - t_imf_start).count();
        
        ++actual_imf_count;
        
        // Accumulate stage times
        total_perturbation_us += stage_perturbation_us;
        total_spline_construct_us += stage_construct_us;
        total_spline_evaluate_us += stage_evaluate_us;
        total_reduction_us += stage_reduction_us;
        total_imf_extract_us += stage_imf_extract_us;
        
        // Decay noise
        noise_amplitude *= config.noise_decay;
        
        // Check stopping criteria
        if (is_monotonic(r_current.data, n, config.monotonic_threshold) ||
            count_extrema(r_current.data, n) < config.min_extrema)
        {
            break;
        }
    }
    
    auto t_decompose_end = Clock::now();
    bd.decompose_total_ms = std::chrono::duration<double, std::milli>(t_decompose_end - t_decompose_start).count();
    
    bd.n_imfs = actual_imf_count;
    bd.n_stages = actual_imf_count;
    
    // Convert to per-stage averages (in ms)
    if (actual_imf_count > 0)
    {
        bd.perturbation_ms = (total_perturbation_us / 1000.0) / actual_imf_count;
        bd.spline_construct_ms = (total_spline_construct_us / 1000.0) / actual_imf_count;
        bd.spline_evaluate_ms = (total_spline_evaluate_us / 1000.0) / actual_imf_count;
        bd.reduction_ms = (total_reduction_us / 1000.0) / actual_imf_count;
        bd.imf_extract_ms = (total_imf_extract_us / 1000.0) / actual_imf_count;
    }
    
    return bd;
}

void bench_cycle_breakdown()
{
    std::cout << "\n=== ICEEMDAN Cycle Breakdown ===\n";
    
    const int32_t n = 2048;
    std::vector<double> signal(n);
    generate_synthetic_signal(signal.data(), n, 0.01);
    
    ICEEMDANConfig config;
    config.max_imfs = 8;
    config.ensemble_size = 100;
    config.noise_std = 0.2;
    
    std::cout << "Signal: " << n << ", Ensemble: " << config.ensemble_size 
              << ", Threads: " << omp_get_max_threads() << "\n";
    
    // Run profiled decomposition
    auto breakdown = profile_iceemdan_decomposition(signal.data(), n, config);
    breakdown.print();
    
    // Compare with optimized parallel version
    std::cout << "\n--- Optimized Parallel Comparison ---\n";
    
    ICEEMDAN decomposer(config);
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    
    // Warmup
    decomposer.decompose(signal.data(), n, imfs, residue);
    
    // Timed
    auto t0 = Clock::now();
    for (int i = 0; i < 5; ++i)
    {
        decomposer.decompose(signal.data(), n, imfs, residue);
    }
    auto t1 = Clock::now();
    double parallel_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
    
    std::cout << "  Sequential profiled:  " << std::setw(8) << std::fixed << std::setprecision(2)
              << breakdown.decompose_total_ms << " ms\n";
    std::cout << "  Parallel optimized:   " << std::setw(8) << parallel_time << " ms\n";
    std::cout << "  Parallel speedup:     " << std::setw(8) << std::setprecision(1)
              << breakdown.decompose_total_ms / parallel_time << "x\n";
}

void bench_scaling_breakdown()
{
    std::cout << "\n=== Scaling Analysis: Where Time Goes ===\n\n";
    
    std::cout << "Signal length scaling (Ensemble=100):\n";
    std::cout << std::setw(8) << "N"
              << std::setw(12) << "NoiseBank"
              << std::setw(12) << "Decompose"
              << std::setw(12) << "Spline%"
              << std::setw(12) << "Perturb%" << "\n";
    std::cout << std::string(56, '-') << "\n";
    
    for (int32_t n : {512, 1024, 2048, 4096})
    {
        std::vector<double> signal(n);
        generate_synthetic_signal(signal.data(), n, 0.01);
        
        ICEEMDANConfig config;
        config.max_imfs = 8;
        config.ensemble_size = 100;
        config.noise_std = 0.2;
        
        auto bd = profile_iceemdan_decomposition(signal.data(), n, config);
        
        double spline_pct = 0.0, perturb_pct = 0.0;
        if (bd.decompose_total_ms > 0 && bd.n_stages > 0)
        {
            spline_pct = 100.0 * (bd.spline_construct_ms + bd.spline_evaluate_ms) * bd.n_stages / bd.decompose_total_ms;
            perturb_pct = 100.0 * bd.perturbation_ms * bd.n_stages / bd.decompose_total_ms;
        }
        
        std::cout << std::setw(8) << n
                  << std::setw(12) << std::fixed << std::setprecision(1) << bd.noise_bank_total_ms
                  << std::setw(12) << bd.decompose_total_ms
                  << std::setw(11) << std::setprecision(0) << spline_pct << "%"
                  << std::setw(11) << perturb_pct << "%" << "\n";
    }
    
    std::cout << "\nEnsemble size scaling (N=1024):\n";
    std::cout << std::setw(10) << "Ensemble"
              << std::setw(12) << "NoiseBank"
              << std::setw(12) << "Decompose"
              << std::setw(14) << "ms/trial" << "\n";
    std::cout << std::string(48, '-') << "\n";
    
    const int32_t n = 1024;
    std::vector<double> signal(n);
    generate_synthetic_signal(signal.data(), n, 0.01);
    
    for (int32_t ens : {25, 50, 100, 200})
    {
        ICEEMDANConfig config;
        config.max_imfs = 8;
        config.ensemble_size = ens;
        config.noise_std = 0.2;
        
        auto bd = profile_iceemdan_decomposition(signal.data(), n, config);
        
        std::cout << std::setw(10) << ens
                  << std::setw(12) << std::fixed << std::setprecision(1) << bd.noise_bank_total_ms
                  << std::setw(12) << bd.decompose_total_ms
                  << std::setw(14) << std::setprecision(2) << bd.decompose_total_ms / ens << "\n";
    }
}

void bench_spline_deep_dive()
{
    std::cout << "\n=== Spline Performance Deep Dive ===\n\n";
    
    const int32_t n = 2048;
    
    // Create realistic extrema pattern
    std::vector<double> signal(n);
    generate_synthetic_signal(signal.data(), n, 0.01);
    
    // Find extrema
    std::vector<int32_t> max_idx(n / 2 + 2);
    std::vector<int32_t> min_idx(n / 2 + 2);
    int32_t n_max, n_min;
    
    find_maxima_raw(signal.data(), n, max_idx.data(), n_max);
    find_minima_raw(signal.data(), n, min_idx.data(), n_min);
    
    std::cout << "Signal: " << n << " samples, " << n_max << " maxima, " << n_min << " minima\n\n";
    
    // Extend extrema
    std::vector<double> ext_x(n + 20);
    std::vector<double> ext_y(n + 20);
    int32_t n_ext, ext_start;
    
    extend_extrema_raw(max_idx.data(), n_max, signal.data(), n, 2,
                       ext_x.data(), ext_y.data(), n_ext, ext_start);
    
    std::cout << "Extended knots: " << n_ext << "\n\n";
    
    // Benchmark spline operations
    const int32_t n_trials = 1000;
    MKLSpline spline;
    AlignedBuffer<double> result(n);
    
    // Construction
    auto t0 = Clock::now();
    for (int32_t t = 0; t < n_trials; ++t)
    {
        spline.construct(ext_x.data(), ext_y.data(), n_ext);
    }
    auto t1 = Clock::now();
    double construct_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / n_trials;
    
    // Evaluation (uniform)
    spline.construct(ext_x.data(), ext_y.data(), n_ext);
    t0 = Clock::now();
    for (int32_t t = 0; t < n_trials; ++t)
    {
        spline.evaluate_uniform(result.data, n);
    }
    t1 = Clock::now();
    double eval_uniform_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / n_trials;
    
    // Evaluation (sorted - for comparison)
    std::vector<double> sites(n);
    for (int32_t i = 0; i < n; ++i)
    {
        sites[i] = static_cast<double>(i);
    }
    
    t0 = Clock::now();
    for (int32_t t = 0; t < n_trials; ++t)
    {
        spline.evaluate(sites.data(), result.data, n);
    }
    t1 = Clock::now();
    double eval_sorted_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / n_trials;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Operation             Time (μs)    Throughput\n";
    std::cout << std::string(48, '-') << "\n";
    std::cout << "Construct (" << n_ext << " knots)   " << std::setw(8) << construct_us 
              << "    " << std::setprecision(1) << 1e6 / construct_us << " splines/s\n";
    std::cout << std::setprecision(2);
    std::cout << "Eval uniform (" << n << ")     " << std::setw(8) << eval_uniform_us 
              << "    " << std::setprecision(1) << n / eval_uniform_us << " Msamples/s\n";
    std::cout << std::setprecision(2);
    std::cout << "Eval sorted (" << n << ")      " << std::setw(8) << eval_sorted_us 
              << "    " << std::setprecision(1) << n / eval_sorted_us << " Msamples/s\n";
    
    std::cout << "\nUniform speedup: " << std::setprecision(2) << eval_sorted_us / eval_uniform_us << "x\n";
    std::cout << "(DF_UNIFORM_PARTITION enables O(1) knot lookup vs O(log K))\n";
    
    // Per-ICEEMDAN-stage estimate
    std::cout << "\n--- Per-ICEEMDAN-Stage Estimate ---\n";
    int32_t ensemble_size = 100;
    double splines_per_stage = 2.0 * ensemble_size; // upper + lower per trial
    double construct_per_stage = construct_us * splines_per_stage / 1000.0;
    double eval_per_stage = eval_uniform_us * splines_per_stage / 1000.0;
    
    std::cout << "Ensemble: " << ensemble_size << " trials\n";
    std::cout << "Splines per stage: " << static_cast<int>(splines_per_stage) << " (2 per trial)\n";
    std::cout << "Construction: " << std::setprecision(1) << construct_per_stage << " ms/stage\n";
    std::cout << "Evaluation:   " << eval_per_stage << " ms/stage\n";
    std::cout << "Total spline: " << construct_per_stage + eval_per_stage << " ms/stage\n";
}

void bench_financial_signal()
{
    std::cout << "\n=== Financial Signal Test ===\n";
    std::cout << "Simulated regime-switching returns\n\n";
    
    const int32_t n = 4096; // ~1 month of 5-min bars
    std::vector<double> signal(n);
    generate_financial_signal(signal.data(), n, 12345);
    
    ICEEMDANConfig config;
    config.max_imfs = 10;
    config.ensemble_size = 100;
    config.noise_std = 0.2;
    
    ICEEMDAN decomposer(config);
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    
    auto t0 = Clock::now();
    decomposer.decompose(signal.data(), n, imfs, residue);
    auto t1 = Clock::now();
    
    double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    std::cout << "Signal length:      " << n << " samples\n";
    std::cout << "Decomposition time: " << std::fixed << std::setprecision(1) << time_ms << " ms\n";
    std::cout << "IMFs extracted:     " << imfs.size() << "\n\n";
    
    // Analyze IMFs for trading relevance
    std::cout << "--- Trading-Relevant Analysis ---\n";
    std::cout << std::setw(6) << "IMF"
              << std::setw(10) << "Hurst"
              << std::setw(12) << "Spec.Ent"
              << std::setw(12) << "Energy%"
              << std::setw(15) << "Interpretation" << "\n";
    std::cout << std::string(55, '-') << "\n";
    
    // Total energy
    double total_energy = 0.0;
    for (const auto &imf : imfs)
    {
        for (int32_t i = 0; i < n; ++i)
        {
            total_energy += imf[i] * imf[i];
        }
    }
    for (int32_t i = 0; i < n; ++i)
    {
        total_energy += residue[i] * residue[i];
    }
    
    for (size_t i = 0; i < imfs.size(); ++i)
    {
        auto a = analyze_imf(imfs[i].data(), n, 1.0 / 300.0); // 5-min bars
        double energy_pct = 100.0 * a.energy / total_energy;
        
        std::string interpretation;
        if (a.hurst > 0.6)
            interpretation = "TREND";
        else if (a.hurst < 0.4)
            interpretation = "MEAN-REV";
        else if (a.spectral_entropy > 0.85)
            interpretation = "NOISE";
        else
            interpretation = "STRUCTURE";
        
        std::cout << std::setw(6) << i
                  << std::setw(10) << std::setprecision(3) << a.hurst
                  << std::setw(12) << std::setprecision(3) << a.spectral_entropy
                  << std::setw(12) << std::setprecision(1) << energy_pct
                  << std::setw(15) << interpretation << "\n";
    }
    
    // Residue (trend)
    double res_energy = 0.0;
    for (int32_t i = 0; i < n; ++i)
    {
        res_energy += residue[i] * residue[i];
    }
    double res_hurst = estimate_hurst_rs(residue.data(), n);
    
    std::cout << std::setw(6) << "RES"
              << std::setw(10) << std::setprecision(3) << res_hurst
              << std::setw(12) << "-"
              << std::setw(12) << std::setprecision(1) << 100.0 * res_energy / total_energy
              << std::setw(15) << "TREND" << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    eemd_init_low_latency(8, true);
    
    std::cout << "\nICEEMDAN-MKL Performance Benchmark\n";
    std::cout << "===================================\n";
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";
    std::cout << "MKL threads: " << mkl_get_max_threads() << " (sequential for ICEEMDAN)\n";
    
    bench_signal_length();
    bench_ensemble_size();
    bench_thread_scaling();
    bench_eemd_vs_iceemdan();
    
    // Detailed cycle breakdown
    bench_cycle_breakdown();
    bench_scaling_breakdown();
    bench_spline_deep_dive();
    
    bench_analysis_utilities();
    bench_financial_signal();
    
    std::cout << "\nBenchmark complete.\n";
    return 0;
}