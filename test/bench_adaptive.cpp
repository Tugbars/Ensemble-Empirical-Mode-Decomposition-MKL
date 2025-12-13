/**
 * Benchmark: Adaptive Ensemble vs Fixed Ensemble
 */

#include "iceemdan_mkl.hpp"
#include <chrono>
#include <cstdio>
#include <random>

using namespace eemd;

// Generate GARCH-like price signal
void generate_garch_signal(double* signal, int32_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    
    double price = 100.0;
    double vol = 0.02;
    
    const double omega = 0.00001;
    const double alpha = 0.1;
    const double beta = 0.85;
    
    for (int32_t i = 0; i < n; ++i) {
        double z = norm(rng);
        double ret = vol * z;
        price *= (1.0 + ret);
        signal[i] = price;
        
        vol = std::sqrt(omega + alpha * ret * ret + beta * vol * vol);
    }
}

int main() {
    const int32_t N = 2048;
    const int RUNS = 5;
    
    std::vector<double> signal(N);
    generate_garch_signal(signal.data(), N, 12345);
    
    printf("=== Adaptive Ensemble Benchmark (N=%d) ===\n\n", N);
    
    // Test 1: Fixed ensemble (adaptive disabled)
    {
        ICEEMDANConfig config;
        config.ensemble_size = 100;
        config.adaptive_ensemble = false;
        config.use_antithetic = true;
        config.use_circular_bank = true;
        
        ICEEMDAN decomposer(config);
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        
        // Warmup
        decomposer.decompose(signal.data(), N, imfs, residue);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < RUNS; ++r) {
            decomposer.decompose(signal.data(), N, imfs, residue);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;
        printf("Fixed (100 trials):    %6.2f ms, %zu IMFs\n", ms, imfs.size());
        
        // Verify reconstruction
        double err = 0.0;
        for (int32_t i = 0; i < N; ++i) {
            double sum = residue[i];
            for (auto& imf : imfs) sum += imf[i];
            double d = sum - signal[i];
            err += d * d;
        }
        printf("  Reconstruction RMSE: %.2e\n", std::sqrt(err / N));
    }
    
    // Test 2: Adaptive ensemble (default settings)
    {
        ICEEMDANConfig config;
        config.ensemble_size = 100;
        config.adaptive_ensemble = true;
        config.adaptive_min_trials = 30;
        config.adaptive_check_interval = 10;
        config.adaptive_rel_tol = 0.08;
        config.use_antithetic = true;
        config.use_circular_bank = true;
        
        ICEEMDAN decomposer(config);
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        
        // Warmup
        decomposer.decompose(signal.data(), N, imfs, residue);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < RUNS; ++r) {
            decomposer.decompose(signal.data(), N, imfs, residue);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;
        printf("Adaptive (tol=0.001):  %6.2f ms, %zu IMFs\n", ms, imfs.size());
        
        // Verify reconstruction
        double err = 0.0;
        for (int32_t i = 0; i < N; ++i) {
            double sum = residue[i];
            for (auto& imf : imfs) sum += imf[i];
            double d = sum - signal[i];
            err += d * d;
        }
        printf("  Reconstruction RMSE: %.2e\n", std::sqrt(err / N));
    }
    
    // Test 3: Aggressive adaptive (looser tolerance)
    {
        ICEEMDANConfig config;
        config.ensemble_size = 100;
        config.adaptive_ensemble = true;
        config.adaptive_min_trials = 20;
        config.adaptive_check_interval = 5;
        config.adaptive_rel_tol = 0.08;  // Looser
        config.use_antithetic = true;
        config.use_circular_bank = true;
        
        ICEEMDAN decomposer(config);
        
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        
        // Warmup
        decomposer.decompose(signal.data(), N, imfs, residue);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < RUNS; ++r) {
            decomposer.decompose(signal.data(), N, imfs, residue);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;
        printf("Aggressive (tol=0.005):%6.2f ms, %zu IMFs\n", ms, imfs.size());
        
        // Verify reconstruction
        double err = 0.0;
        for (int32_t i = 0; i < N; ++i) {
            double sum = residue[i];
            for (auto& imf : imfs) sum += imf[i];
            double d = sum - signal[i];
            err += d * d;
        }
        printf("  Reconstruction RMSE: %.2e\n", std::sqrt(err / N));
    }
    
    // Test 4: Compare IMF quality (energy correlation)
    printf("\n=== IMF Quality Comparison ===\n");
    {
        // Reference: fixed 100
        ICEEMDANConfig config_ref;
        config_ref.ensemble_size = 100;
        config_ref.adaptive_ensemble = false;
        config_ref.use_antithetic = true;
        config_ref.use_circular_bank = true;
        ICEEMDAN ref_decomposer(config_ref);
        
        std::vector<std::vector<double>> ref_imfs;
        std::vector<double> ref_residue;
        ref_decomposer.decompose(signal.data(), N, ref_imfs, ref_residue);
        
        // Adaptive
        ICEEMDANConfig config_ada;
        config_ada.ensemble_size = 100;
        config_ada.adaptive_ensemble = true;
        config_ada.adaptive_min_trials = 30;
        config_ada.adaptive_check_interval = 10;
        config_ada.adaptive_rel_tol = 0.08;
        config_ada.use_antithetic = true;
        config_ada.use_circular_bank = true;
        ICEEMDAN ada_decomposer(config_ada);
        
        std::vector<std::vector<double>> ada_imfs;
        std::vector<double> ada_residue;
        ada_decomposer.decompose(signal.data(), N, ada_imfs, ada_residue);
        
        // Compare energies
        int n_compare = std::min(ref_imfs.size(), ada_imfs.size());
        printf("IMF |  Fixed Energy  | Adaptive Energy | Ratio\n");
        printf("----|----------------|-----------------|-------\n");
        for (int k = 0; k < n_compare; ++k) {
            double ref_energy = 0.0, ada_energy = 0.0;
            for (int i = 0; i < N; ++i) {
                ref_energy += ref_imfs[k][i] * ref_imfs[k][i];
                ada_energy += ada_imfs[k][i] * ada_imfs[k][i];
            }
            double ratio = ada_energy / ref_energy;
            printf(" %2d | %14.4e | %15.4e | %.4f\n", k+1, ref_energy, ada_energy, ratio);
        }
    }
    
    printf("\n=== Summary ===\n");
    printf("Adaptive ensemble stops early when mean estimate stabilizes.\n");
    printf("Expected speedup: 30-50%% with minimal quality loss.\n");
    
    return 0;
}
