/**
 * ICEEMDAN Finance Mode Benchmark
 * 
 * Tests finance-specific features:
 * - ProcessingMode comparison (Standard vs Finance vs Scientific)
 * - Volatility methods (Global vs SMA vs EMA)
 * - Boundary methods (Mirror vs AR vs Linear)
 * - Regime change handling (volatility clustering)
 * - NaN/Inf sanitization overhead
 * - Diagnostics overhead
 * - Right-edge causality validation
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
#include <random>
#include <numeric>

using namespace eemd;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Finance Test Signals
// ============================================================================

/**
 * Generate realistic price series with volatility clustering (GARCH-like)
 * Simulates regime switches between low and high volatility periods
 */
void generate_garch_price(double *signal, int32_t n, uint32_t seed)
{
    VSLStreamStatePtr stream = nullptr;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    
    std::vector<double> innovations(n);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, innovations.data(), 0.0, 1.0);
    
    // GARCH(1,1) parameters
    const double omega = 0.00001;
    const double alpha = 0.1;
    const double beta = 0.85;
    
    double sigma2 = 0.0001;  // Initial variance
    signal[0] = 100.0;       // Initial price
    
    for (int32_t i = 1; i < n; ++i)
    {
        // Update variance (GARCH)
        double eps_prev = innovations[i-1] * std::sqrt(sigma2);
        sigma2 = omega + alpha * eps_prev * eps_prev + beta * sigma2;
        
        // Log return
        double ret = std::sqrt(sigma2) * innovations[i];
        
        // Price update
        signal[i] = signal[i-1] * std::exp(ret);
    }
    
    vslDeleteStream(&stream);
}

/**
 * Generate signal with explicit regime changes
 * Low vol (0.5%) → High vol (3%) → Low vol
 */
void generate_regime_switch(double *signal, int32_t n, uint32_t seed)
{
    VSLStreamStatePtr stream = nullptr;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    
    std::vector<double> noise(n);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, noise.data(), 0.0, 1.0);
    
    signal[0] = 100.0;
    
    for (int32_t i = 1; i < n; ++i)
    {
        double vol;
        if (i < n / 3)
            vol = 0.005;           // Low vol regime
        else if (i < 2 * n / 3)
            vol = 0.03;            // High vol regime (6x)
        else
            vol = 0.005;           // Back to low vol
        
        signal[i] = signal[i-1] * (1.0 + vol * noise[i]);
    }
    
    vslDeleteStream(&stream);
}

/**
 * Generate signal with NaN/Inf (simulates bad data feed)
 */
void generate_dirty_signal(double *signal, int32_t n, uint32_t seed, double bad_ratio = 0.01)
{
    generate_garch_price(signal, n, seed);
    
    std::mt19937 rng(seed + 1);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    std::uniform_int_distribution<> type_dist(0, 2);
    
    int32_t bad_count = 0;
    for (int32_t i = 0; i < n; ++i)
    {
        if (dist(rng) < bad_ratio)
        {
            switch (type_dist(rng))
            {
                case 0: signal[i] = std::nan(""); break;
                case 1: signal[i] = std::numeric_limits<double>::infinity(); break;
                case 2: signal[i] = -std::numeric_limits<double>::infinity(); break;
            }
            ++bad_count;
        }
    }
    
    std::cout << "    (Injected " << bad_count << " bad values, " 
              << std::fixed << std::setprecision(1) << (100.0 * bad_count / n) << "%)\n";
}

/**
 * Generate trending signal with flash crash at right edge
 * Tests AR boundary extrapolation vs mirroring
 */
void generate_flash_crash(double *signal, int32_t n, uint32_t seed)
{
    VSLStreamStatePtr stream = nullptr;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    
    std::vector<double> noise(n);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, noise.data(), 0.0, 1.0);
    
    signal[0] = 100.0;
    
    for (int32_t i = 1; i < n; ++i)
    {
        double vol = 0.005;
        double drift = 0.0001;  // Slight uptrend
        
        // Flash crash in last 5%
        if (i > 0.95 * n)
        {
            vol = 0.05;
            drift = -0.01;  // Strong downtrend
        }
        
        signal[i] = signal[i-1] * (1.0 + drift + vol * noise[i]);
    }
    
    vslDeleteStream(&stream);
}

// ============================================================================
// Benchmark Utilities
// ============================================================================

struct FinanceBenchResult
{
    double time_ms;
    int32_t n_imfs;
    double orthogonality;
    int32_t nan_count;
    double right_edge_slope;  // For causality check
};

double compute_right_edge_slope(const std::vector<double> &imf, int32_t lookback = 10)
{
    if (imf.size() < static_cast<size_t>(lookback + 1)) return 0.0;
    
    int32_t n = static_cast<int32_t>(imf.size());
    double sum_slope = 0.0;
    
    for (int32_t i = n - lookback; i < n; ++i)
    {
        sum_slope += imf[i] - imf[i-1];
    }
    
    return sum_slope / lookback;
}

const char* mode_name(ProcessingMode mode)
{
    switch (mode)
    {
        case ProcessingMode::Standard: return "Standard";
        case ProcessingMode::Finance: return "Finance";
        case ProcessingMode::Scientific: return "Scientific";
        default: return "Unknown";
    }
}

const char* vol_method_name(VolatilityMethod method)
{
    switch (method)
    {
        case VolatilityMethod::Global: return "Global";
        case VolatilityMethod::SMA: return "SMA";
        case VolatilityMethod::EMA: return "EMA";
        default: return "Unknown";
    }
}

const char* boundary_name(BoundaryMethod method)
{
    switch (method)
    {
        case BoundaryMethod::Mirror: return "Mirror";
        case BoundaryMethod::AR: return "AR(1)";
        case BoundaryMethod::Linear: return "Linear";
        default: return "Unknown";
    }
}

// ============================================================================
// Benchmark: Processing Mode Comparison
// ============================================================================

void bench_processing_modes()
{
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "BENCHMARK: Processing Mode Comparison\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    const int32_t n = 4096;
    const int32_t n_trials = 5;
    
    std::vector<double> signal(n);
    generate_garch_price(signal.data(), n, 42);
    
    std::cout << "Signal: GARCH price, N=" << n << ", Trials=" << n_trials << "\n\n";
    
    std::cout << std::setw(12) << "Mode"
              << std::setw(12) << "Time (ms)"
              << std::setw(10) << "IMFs"
              << std::setw(12) << "Ortho Idx"
              << std::setw(15) << "Volatility"
              << std::setw(12) << "Boundary" << "\n";
    std::cout << std::string(73, '-') << "\n";
    
    for (ProcessingMode mode : {ProcessingMode::Standard, ProcessingMode::Finance, ProcessingMode::Scientific})
    {
        ICEEMDAN decomposer(mode);
        decomposer.config().ensemble_size = 50;  // Faster for benchmark
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        DecompositionDiagnostics diag;
        
        // Warmup
        decomposer.decompose(signal.data(), n, imfs, residue);
        
        // Timed runs
        double total_time = 0.0;
        for (int32_t t = 0; t < n_trials; ++t)
        {
            auto t0 = Clock::now();
            decomposer.decompose_with_diagnostics(signal.data(), n, imfs, residue, diag);
            auto t1 = Clock::now();
            total_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        
        double avg_time = total_time / n_trials;
        
        std::cout << std::setw(12) << mode_name(mode)
                  << std::setw(12) << std::fixed << std::setprecision(2) << avg_time
                  << std::setw(10) << imfs.size()
                  << std::setw(12) << std::setprecision(4) << diag.orthogonality_index
                  << std::setw(15) << vol_method_name(decomposer.config().volatility_method)
                  << std::setw(12) << boundary_name(decomposer.config().boundary_method) << "\n";
    }
}

// ============================================================================
// Benchmark: Volatility Method Comparison
// ============================================================================

void bench_volatility_methods()
{
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "BENCHMARK: Volatility Method Comparison (Regime-Switching Signal)\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    const int32_t n = 4096;
    const int32_t n_trials = 5;
    
    std::vector<double> signal(n);
    generate_regime_switch(signal.data(), n, 42);
    
    std::cout << "Signal: Regime switch (low→high→low vol), N=" << n << "\n\n";
    
    std::cout << std::setw(10) << "Method"
              << std::setw(10) << "Window"
              << std::setw(12) << "Time (ms)"
              << std::setw(10) << "IMFs"
              << std::setw(14) << "Ortho Idx" << "\n";
    std::cout << std::string(56, '-') << "\n";
    
    // Global
    {
        ICEEMDAN decomposer;
        decomposer.config().volatility_method = VolatilityMethod::Global;
        decomposer.config().ensemble_size = 50;
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        DecompositionDiagnostics diag;
        
        decomposer.decompose(signal.data(), n, imfs, residue);  // Warmup
        
        double total_time = 0.0;
        for (int32_t t = 0; t < n_trials; ++t)
        {
            auto t0 = Clock::now();
            decomposer.decompose_with_diagnostics(signal.data(), n, imfs, residue, diag);
            auto t1 = Clock::now();
            total_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        
        std::cout << std::setw(10) << "Global"
                  << std::setw(10) << "-"
                  << std::setw(12) << std::fixed << std::setprecision(2) << total_time / n_trials
                  << std::setw(10) << imfs.size()
                  << std::setw(14) << std::setprecision(4) << diag.orthogonality_index << "\n";
    }
    
    // SMA with different windows
    for (int32_t window : {20, 50, 100})
    {
        ICEEMDAN decomposer;
        decomposer.config().volatility_method = VolatilityMethod::SMA;
        decomposer.config().vol_window = window;
        decomposer.config().ensemble_size = 50;
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        DecompositionDiagnostics diag;
        
        decomposer.decompose(signal.data(), n, imfs, residue);
        
        double total_time = 0.0;
        for (int32_t t = 0; t < n_trials; ++t)
        {
            auto t0 = Clock::now();
            decomposer.decompose_with_diagnostics(signal.data(), n, imfs, residue, diag);
            auto t1 = Clock::now();
            total_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        
        std::cout << std::setw(10) << "SMA"
                  << std::setw(10) << window
                  << std::setw(12) << std::fixed << std::setprecision(2) << total_time / n_trials
                  << std::setw(10) << imfs.size()
                  << std::setw(14) << std::setprecision(4) << diag.orthogonality_index << "\n";
    }
    
    // EMA with different spans
    for (int32_t span : {10, 20, 50})
    {
        ICEEMDAN decomposer;
        decomposer.config().volatility_method = VolatilityMethod::EMA;
        decomposer.config().vol_ema_span = span;
        decomposer.config().ensemble_size = 50;
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        DecompositionDiagnostics diag;
        
        decomposer.decompose(signal.data(), n, imfs, residue);
        
        double total_time = 0.0;
        for (int32_t t = 0; t < n_trials; ++t)
        {
            auto t0 = Clock::now();
            decomposer.decompose_with_diagnostics(signal.data(), n, imfs, residue, diag);
            auto t1 = Clock::now();
            total_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        
        std::cout << std::setw(10) << "EMA"
                  << std::setw(10) << span
                  << std::setw(12) << std::fixed << std::setprecision(2) << total_time / n_trials
                  << std::setw(10) << imfs.size()
                  << std::setw(14) << std::setprecision(4) << diag.orthogonality_index << "\n";
    }
}

// ============================================================================
// Benchmark: Boundary Method Comparison (Flash Crash)
// ============================================================================

void bench_boundary_methods()
{
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "BENCHMARK: Boundary Method Comparison (Flash Crash Signal)\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    const int32_t n = 2048;
    const int32_t n_trials = 5;
    
    std::vector<double> signal(n);
    generate_flash_crash(signal.data(), n, 42);
    
    std::cout << "Signal: Uptrend with flash crash at right edge, N=" << n << "\n";
    std::cout << "Checking right-edge behavior (negative slope = crash detected)\n\n";
    
    std::cout << std::setw(10) << "Method"
              << std::setw(12) << "Time (ms)"
              << std::setw(10) << "IMFs"
              << std::setw(16) << "IMF1 R-Slope"
              << std::setw(16) << "IMF2 R-Slope" << "\n";
    std::cout << std::string(64, '-') << "\n";
    
    for (BoundaryMethod method : {BoundaryMethod::Mirror, BoundaryMethod::AR, BoundaryMethod::Linear})
    {
        ICEEMDAN decomposer;
        decomposer.config().boundary_method = method;
        decomposer.config().ensemble_size = 50;
        decomposer.config().ar_damping = 0.5;
        decomposer.config().ar_max_slope_atr = 2.0;
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        
        decomposer.decompose(signal.data(), n, imfs, residue);  // Warmup
        
        double total_time = 0.0;
        for (int32_t t = 0; t < n_trials; ++t)
        {
            auto t0 = Clock::now();
            decomposer.decompose(signal.data(), n, imfs, residue);
            auto t1 = Clock::now();
            total_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        
        double slope1 = imfs.size() > 0 ? compute_right_edge_slope(imfs[0]) : 0.0;
        double slope2 = imfs.size() > 1 ? compute_right_edge_slope(imfs[1]) : 0.0;
        
        std::cout << std::setw(10) << boundary_name(method)
                  << std::setw(12) << std::fixed << std::setprecision(2) << total_time / n_trials
                  << std::setw(10) << imfs.size()
                  << std::setw(16) << std::setprecision(6) << slope1
                  << std::setw(16) << std::setprecision(6) << slope2 << "\n";
    }
    
    std::cout << "\n* Negative slope indicates crash is preserved (good for trading)\n";
    std::cout << "* Mirror may show positive slope (false reversal signal)\n";
}

// ============================================================================
// Benchmark: NaN/Inf Sanitization
// ============================================================================

void bench_sanitization()
{
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "BENCHMARK: NaN/Inf Sanitization Overhead\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    const int32_t n = 4096;
    const int32_t n_trials = 5;
    
    // Clean signal
    std::vector<double> clean_signal(n);
    generate_garch_price(clean_signal.data(), n, 42);
    
    // Dirty signal
    std::vector<double> dirty_signal(n);
    std::cout << "Generating dirty signal with 1% bad values:\n";
    generate_dirty_signal(dirty_signal.data(), n, 42, 0.01);
    
    std::cout << "\n";
    std::cout << std::setw(20) << "Configuration"
              << std::setw(12) << "Time (ms)"
              << std::setw(10) << "IMFs"
              << std::setw(12) << "NaN Count" << "\n";
    std::cout << std::string(54, '-') << "\n";
    
    // Clean signal, no sanitization
    {
        ICEEMDAN decomposer;
        decomposer.config().sanitize_input = false;
        decomposer.config().ensemble_size = 50;
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        
        decomposer.decompose(clean_signal.data(), n, imfs, residue);
        
        double total_time = 0.0;
        for (int32_t t = 0; t < n_trials; ++t)
        {
            auto t0 = Clock::now();
            decomposer.decompose(clean_signal.data(), n, imfs, residue);
            auto t1 = Clock::now();
            total_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        
        std::cout << std::setw(20) << "Clean, no sanitize"
                  << std::setw(12) << std::fixed << std::setprecision(2) << total_time / n_trials
                  << std::setw(10) << imfs.size()
                  << std::setw(12) << "N/A" << "\n";
    }
    
    // Clean signal, with sanitization
    {
        ICEEMDAN decomposer;
        decomposer.config().sanitize_input = true;
        decomposer.config().ensemble_size = 50;
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        DecompositionDiagnostics diag;
        
        decomposer.decompose(clean_signal.data(), n, imfs, residue);
        
        double total_time = 0.0;
        for (int32_t t = 0; t < n_trials; ++t)
        {
            auto t0 = Clock::now();
            decomposer.decompose_with_diagnostics(clean_signal.data(), n, imfs, residue, diag);
            auto t1 = Clock::now();
            total_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        
        std::cout << std::setw(20) << "Clean, sanitize ON"
                  << std::setw(12) << std::fixed << std::setprecision(2) << total_time / n_trials
                  << std::setw(10) << imfs.size()
                  << std::setw(12) << diag.nan_count << "\n";
    }
    
    // Dirty signal, with sanitization (Finance mode)
    {
        ICEEMDAN decomposer(ProcessingMode::Finance);
        decomposer.config().ensemble_size = 50;
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        DecompositionDiagnostics diag;
        
        // This should NOT crash
        decomposer.decompose(dirty_signal.data(), n, imfs, residue);
        
        double total_time = 0.0;
        for (int32_t t = 0; t < n_trials; ++t)
        {
            auto t0 = Clock::now();
            decomposer.decompose_with_diagnostics(dirty_signal.data(), n, imfs, residue, diag);
            auto t1 = Clock::now();
            total_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        
        std::cout << std::setw(20) << "Dirty, Finance mode"
                  << std::setw(12) << std::fixed << std::setprecision(2) << total_time / n_trials
                  << std::setw(10) << imfs.size()
                  << std::setw(12) << diag.nan_count << "\n";
    }
}

// ============================================================================
// Benchmark: Scalar vs Array Volatility (Memory Bandwidth)
// ============================================================================

void bench_scalar_vs_array_vol()
{
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "BENCHMARK: Scalar vs Array Volatility (Memory Bandwidth)\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    std::cout << std::setw(10) << "N"
              << std::setw(15) << "Global (ms)"
              << std::setw(15) << "EMA (ms)"
              << std::setw(12) << "Overhead" << "\n";
    std::cout << std::string(52, '-') << "\n";
    
    for (int32_t n : {2048, 8192, 32768, 131072})
    {
        std::vector<double> signal(n);
        generate_garch_price(signal.data(), n, 42);
        
        const int32_t n_trials = 3;
        
        // Global (scalar path)
        double global_time;
        {
            ICEEMDAN decomposer;
            decomposer.config().volatility_method = VolatilityMethod::Global;
            decomposer.config().ensemble_size = 30;
            
            std::vector<std::vector<double>> imfs;
            std::vector<double> residue;
            
            decomposer.decompose(signal.data(), n, imfs, residue);
            
            auto t0 = Clock::now();
            for (int32_t t = 0; t < n_trials; ++t)
            {
                decomposer.decompose(signal.data(), n, imfs, residue);
            }
            auto t1 = Clock::now();
            global_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_trials;
        }
        
        // EMA (array path)
        double ema_time;
        {
            ICEEMDAN decomposer;
            decomposer.config().volatility_method = VolatilityMethod::EMA;
            decomposer.config().vol_ema_span = 20;
            decomposer.config().ensemble_size = 30;
            
            std::vector<std::vector<double>> imfs;
            std::vector<double> residue;
            
            decomposer.decompose(signal.data(), n, imfs, residue);
            
            auto t0 = Clock::now();
            for (int32_t t = 0; t < n_trials; ++t)
            {
                decomposer.decompose(signal.data(), n, imfs, residue);
            }
            auto t1 = Clock::now();
            ema_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_trials;
        }
        
        double overhead = (ema_time - global_time) / global_time * 100.0;
        
        std::cout << std::setw(10) << n
                  << std::setw(15) << std::fixed << std::setprecision(2) << global_time
                  << std::setw(15) << std::setprecision(2) << ema_time
                  << std::setw(11) << std::setprecision(1) << overhead << "%" << "\n";
    }
    
    std::cout << "\n* Lower overhead = scalar path working (no wasted array writes)\n";
}

// ============================================================================
// Benchmark: AR Damping Sensitivity
// ============================================================================

void bench_ar_damping()
{
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "BENCHMARK: AR Damping Sensitivity (Flash Crash)\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    const int32_t n = 2048;
    
    std::vector<double> signal(n);
    generate_flash_crash(signal.data(), n, 42);
    
    std::cout << "Testing AR damping factor on flash crash signal\n";
    std::cout << "damping=0 → mean revert, damping=1 → full AR extrapolation\n\n";
    
    std::cout << std::setw(10) << "Damping"
              << std::setw(16) << "IMF1 R-Slope"
              << std::setw(16) << "IMF2 R-Slope"
              << std::setw(16) << "Residue Slope" << "\n";
    std::cout << std::string(58, '-') << "\n";
    
    for (double damping : {0.0, 0.25, 0.5, 0.75, 1.0})
    {
        ICEEMDAN decomposer;
        decomposer.config().boundary_method = BoundaryMethod::AR;
        decomposer.config().ar_damping = damping;
        decomposer.config().ar_max_slope_atr = 3.0;
        decomposer.config().ensemble_size = 50;
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        
        decomposer.decompose(signal.data(), n, imfs, residue);
        
        double slope1 = imfs.size() > 0 ? compute_right_edge_slope(imfs[0]) : 0.0;
        double slope2 = imfs.size() > 1 ? compute_right_edge_slope(imfs[1]) : 0.0;
        double slope_res = compute_right_edge_slope(residue);
        
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << damping
                  << std::setw(16) << std::setprecision(6) << slope1
                  << std::setw(16) << std::setprecision(6) << slope2
                  << std::setw(16) << std::setprecision(6) << slope_res << "\n";
    }
    
    std::cout << "\n* Moderate damping (0.5) balances crash detection vs overshoot\n";
}

// ============================================================================
// Benchmark: Full Finance Pipeline
// ============================================================================

void bench_full_finance_pipeline()
{
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "BENCHMARK: Full Finance Pipeline (Production Config)\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    const int32_t n_trials = 10;
    
    std::cout << "Config: Finance mode, Ensemble=50, EMA(20), AR(0.5)\n\n";
    
    std::cout << std::setw(10) << "N"
              << std::setw(14) << "Time (ms)"
              << std::setw(10) << "IMFs"
              << std::setw(14) << "Ortho Idx"
              << std::setw(16) << "Throughput" << "\n";
    std::cout << std::string(64, '-') << "\n";
    
    for (int32_t n : {1024, 2048, 4096, 8192})
    {
        std::vector<double> signal(n);
        generate_garch_price(signal.data(), n, 42);
        
        ICEEMDAN decomposer(ProcessingMode::Finance);
        decomposer.config().ensemble_size = 50;
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        DecompositionDiagnostics diag;
        
        // Warmup
        decomposer.decompose(signal.data(), n, imfs, residue);
        
        auto t0 = Clock::now();
        for (int32_t t = 0; t < n_trials; ++t)
        {
            decomposer.decompose_with_diagnostics(signal.data(), n, imfs, residue, diag);
        }
        auto t1 = Clock::now();
        
        double avg_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_trials;
        double throughput = (50.0 * n / 1e6) / (avg_time / 1000.0);  // MS/s
        
        std::cout << std::setw(10) << n
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_time
                  << std::setw(10) << imfs.size()
                  << std::setw(14) << std::setprecision(4) << diag.orthogonality_index
                  << std::setw(14) << std::setprecision(2) << throughput << " MS/s" << "\n";
    }
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    std::cout << "ICEEMDAN Finance Mode Benchmark\n";
    std::cout << "================================\n";
    std::cout << "Threads: " << omp_get_max_threads() << "\n";
    
    bench_processing_modes();
    bench_volatility_methods();
    bench_boundary_methods();
    bench_sanitization();
    bench_scalar_vs_array_vol();
    bench_ar_damping();
    bench_full_finance_pipeline();
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Benchmark complete.\n";
    
    return 0;
}
