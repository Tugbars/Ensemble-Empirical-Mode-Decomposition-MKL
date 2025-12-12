/**
 * ICEEMDAN (Improved Complete Ensemble EMD with Adaptive Noise)
 *
 * Extension of EEMD-MKL implementation.
 *
 * Key differences from EEMD:
 * - Noise is added at EACH decomposition stage, not just once
 * - We add the IMF of noise (not raw noise) — "adaptive" to current scale
 * - Results in cleaner IMFs with less mode mixing and residual noise
 *
 * Algorithm (Colominas et al., 2014):
 *
 * 1. Pre-decompose M noise realizations to get their IMFs
 * 2. First IMF:
 *    - For each trial i: compute local_mean(x + ε₀·E₁(w^i))
 *    - Average local means → r₁
 *    - IMF₁ = x - r₁
 *
 * 3. For k-th IMF (k ≥ 2):
 *    - For each trial i: compute local_mean(r_{k-1} + ε_{k-1}·E_k(w^i))
 *    - Average local means → r_k
 *    - IMF_k = r_{k-1} - r_k
 *
 * 4. Continue until residue is monotonic or max IMFs reached
 *
 * Reference:
 * Colominas, M. A., Schlotthauer, G., & Torres, M. E. (2014).
 * "Improved complete ensemble EMD: A suitable tool for biomedical signal processing"
 * Biomedical Signal Processing and Control, 14, 19-29.
 *
 * License: MIT
 */

#ifndef ICEEMDAN_MKL_HPP
#define ICEEMDAN_MKL_HPP

#include "eemd_mkl.hpp"

namespace eemd
{

    // ============================================================================
    // ICEEMDAN Configuration
    // ============================================================================

    struct ICEEMDANConfig
    {
        int32_t max_imfs = 10;
        int32_t max_sift_iters = 100;
        double sift_threshold = 0.05;
        int32_t ensemble_size = 100;
        double noise_std = 0.2;           // ε₀ initial noise strength
        double noise_decay = 1.0;         // Multiply noise_std by this each stage (1.0 = no decay)
        int32_t boundary_extend = 2;
        uint32_t rng_seed = 42;
        double monotonic_threshold = 1e-6; // Stop if residue is nearly monotonic
        int32_t min_extrema = 3;           // Minimum extrema to continue decomposition
    };

    // ============================================================================
    // Helper Functions
    // ============================================================================

    /**
     * Check if signal is monotonic (stopping criterion)
     */
    inline bool is_monotonic(const double *signal, int32_t n, double threshold = 1e-6)
    {
        if (n < 3)
            return true;

        int32_t n_increasing = 0;
        int32_t n_decreasing = 0;

        for (int32_t i = 1; i < n; ++i)
        {
            double diff = signal[i] - signal[i - 1];
            if (diff > threshold)
                ++n_increasing;
            else if (diff < -threshold)
                ++n_decreasing;
        }

        // Monotonic if almost all differences have same sign
        double ratio = static_cast<double>(std::max(n_increasing, n_decreasing)) / (n - 1);
        return ratio > 0.95;
    }

    /**
     * Count extrema in signal
     */
    inline int32_t count_extrema(const double *signal, int32_t n)
    {
        if (n < 3)
            return 0;

        int32_t count = 0;
        for (int32_t i = 1; i < n - 1; ++i)
        {
            bool is_max = (signal[i] > signal[i - 1]) && (signal[i] > signal[i + 1]);
            bool is_min = (signal[i] < signal[i - 1]) && (signal[i] < signal[i + 1]);
            if (is_max || is_min)
                ++count;
        }
        return count;
    }

    /**
     * Compute signal standard deviation
     */
    inline double compute_std(const double *signal, int32_t n)
    {
        double mean = 0.0;
        EEMD_OMP_SIMD_REDUCTION(+, mean)
        for (int32_t i = 0; i < n; ++i)
        {
            mean += signal[i];
        }
        mean /= n;

        double var = 0.0;
        EEMD_OMP_SIMD_REDUCTION(+, var)
        for (int32_t i = 0; i < n; ++i)
        {
            double d = signal[i] - mean;
            var += d * d;
        }

        return std::sqrt(var / n);
    }

    // ============================================================================
    // Local Mean Computer
    // Computes the mean of upper and lower envelopes without extracting IMF
    // ============================================================================

    class LocalMeanComputer
    {
    public:
        explicit LocalMeanComputer(int32_t max_len, int32_t boundary_extend)
            : max_len_(max_len), boundary_extend_(boundary_extend), max_idx_(max_len / 2 + 2), min_idx_(max_len / 2 + 2), ext_x_(max_len + 20), ext_y_(max_len + 20), upper_env_(max_len), lower_env_(max_len)
        {
        }

        /**
         * Compute local mean (average of upper and lower envelopes)
         * Returns false if not enough extrema
         */
        bool compute(const double *signal, int32_t n, double *local_mean)
        {
            // Find extrema
            find_maxima_raw(signal, n, max_idx_.data(), n_max_);
            find_minima_raw(signal, n, min_idx_.data(), n_min_);

            if (n_max_ < 2 || n_min_ < 2)
            {
                // Not enough extrema — return signal as local mean (no oscillation)
                std::memcpy(local_mean, signal, n * sizeof(double));
                return false;
            }

            // Upper envelope
            int32_t n_ext, ext_start;
            extend_extrema_raw(max_idx_.data(), n_max_, signal, n,
                               boundary_extend_, ext_x_.data(), ext_y_.data(),
                               n_ext, ext_start);

            if (!upper_spline_.construct(ext_x_.data(), ext_y_.data(), n_ext))
            {
                std::memcpy(local_mean, signal, n * sizeof(double));
                return false;
            }
            if (!upper_spline_.evaluate_uniform(upper_env_.data, n))
            {
                std::memcpy(local_mean, signal, n * sizeof(double));
                return false;
            }

            // Lower envelope
            extend_extrema_raw(min_idx_.data(), n_min_, signal, n,
                               boundary_extend_, ext_x_.data(), ext_y_.data(),
                               n_ext, ext_start);

            if (!lower_spline_.construct(ext_x_.data(), ext_y_.data(), n_ext))
            {
                std::memcpy(local_mean, signal, n * sizeof(double));
                return false;
            }
            if (!lower_spline_.evaluate_uniform(lower_env_.data, n))
            {
                std::memcpy(local_mean, signal, n * sizeof(double));
                return false;
            }

            // Local mean = (upper + lower) / 2
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

        int32_t get_n_extrema() const { return n_max_ + n_min_; }

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

    // ============================================================================
    // Pre-decomposed Noise Bank
    // Stores IMFs of white noise realizations for reuse
    // ============================================================================

    class NoiseBank
    {
    public:
        NoiseBank() = default;

        /**
         * Pre-decompose M realizations of white noise
         * Each noise signal → decomposed into max_imfs IMFs
         */
        void initialize(int32_t n, int32_t ensemble_size, int32_t max_imfs,
                        const EEMDConfig &emd_config, uint32_t base_seed)
        {
            n_ = n;
            ensemble_size_ = ensemble_size;
            max_imfs_ = max_imfs;

            // Allocate: [ensemble_size][max_imfs][n]
            noise_imfs_.resize(ensemble_size);
            imf_counts_.resize(ensemble_size);

            for (int32_t i = 0; i < ensemble_size; ++i)
            {
                noise_imfs_[i].resize(max_imfs);
                for (int32_t k = 0; k < max_imfs; ++k)
                {
                    noise_imfs_[i][k].resize(n);
                }
            }

// Decompose noise realizations in parallel
#pragma omp parallel
            {
                const int32_t tid = omp_get_thread_num();

                // Thread-local RNG
                VSLStreamStatePtr stream = nullptr;
                vslNewStream(&stream, VSL_BRNG_MT19937, base_seed + tid * 10000);

                // Thread-local buffers
                AlignedBuffer<double> noise(n);
                AlignedBuffer<double> work(n);

                Sifter sifter(n, emd_config);

#pragma omp for schedule(dynamic)
                for (int32_t i = 0; i < ensemble_size; ++i)
                {
                    // Generate unit variance white noise
                    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream,
                                  n, noise.data, 0.0, 1.0);

                    // Copy to work buffer
                    std::memcpy(work.data, noise.data, n * sizeof(double));

                    // Decompose
                    int32_t imf_count = 0;
                    for (int32_t k = 0; k < max_imfs; ++k)
                    {
                        if (!sifter.sift_imf(work.data, noise_imfs_[i][k].data(), n))
                        {
                            break;
                        }
                        ++imf_count;
                    }
                    imf_counts_[i] = imf_count;
                }

                vslDeleteStream(&stream);
            }

            initialized_ = true;
        }

        /**
         * Get k-th IMF of i-th noise realization
         * Returns nullptr if k >= number of IMFs for this realization
         */
        const double *get_noise_imf(int32_t trial_idx, int32_t imf_idx) const
        {
            if (trial_idx >= ensemble_size_ || imf_idx >= imf_counts_[trial_idx])
            {
                return nullptr;
            }
            return noise_imfs_[trial_idx][imf_idx].data();
        }

        int32_t get_imf_count(int32_t trial_idx) const
        {
            return imf_counts_[trial_idx];
        }

        bool is_initialized() const { return initialized_; }

    private:
        int32_t n_ = 0;
        int32_t ensemble_size_ = 0;
        int32_t max_imfs_ = 0;
        bool initialized_ = false;

        // [ensemble_size][max_imfs] -> vector<double> of length n
        std::vector<std::vector<std::vector<double>>> noise_imfs_;
        std::vector<int32_t> imf_counts_;
    };

    // ============================================================================
    // ICEEMDAN Main Class
    // ============================================================================

    class ICEEMDAN
    {
    public:
        explicit ICEEMDAN(const ICEEMDANConfig &config = ICEEMDANConfig())
            : config_(config)
        {
        }

        /**
         * Decompose signal into IMFs using ICEEMDAN algorithm
         *
         * HIGH-PERFORMANCE VERSION:
         * - Single fork-join (threads persist across all IMF stages)
         * - MKL spline tasks allocated ONCE per thread
         * - Manual loop chunking (avoids omp for implicit barriers)
         * - Explicit barriers for synchronization
         *
         * @param signal     Input signal
         * @param n          Signal length
         * @param imfs       Output: vector of IMFs
         * @param residue    Output: final residue
         * @return           true on success
         */
        bool decompose(
            const double *signal,
            int32_t n,
            std::vector<std::vector<double>> &imfs,
            std::vector<double> &residue)
        {
            if (n < 4)
                return false;

            // Compute signal statistics
            const double signal_std = compute_std(signal, n);

            // Build EMD config from ICEEMDAN config
            EEMDConfig emd_config;
            emd_config.max_imfs = config_.max_imfs;
            emd_config.max_sift_iters = config_.max_sift_iters;
            emd_config.sift_threshold = config_.sift_threshold;
            emd_config.boundary_extend = config_.boundary_extend;

            // Pre-decompose noise bank
            NoiseBank noise_bank;
            noise_bank.initialize(n, config_.ensemble_size, config_.max_imfs,
                                  emd_config, config_.rng_seed);

            // Prepare output
            imfs.clear();
            imfs.reserve(config_.max_imfs);

            // Shared state across all threads and IMF stages
            AlignedBuffer<double> r_current(n);
            std::memcpy(r_current.data, signal, n * sizeof(double));

            AlignedBuffer<double> mean_accumulator(n);
            mean_accumulator.zero();

            // Pre-allocate IMF storage (avoid realloc inside parallel region)
            std::vector<AlignedBuffer<double>> imf_storage(config_.max_imfs);
            for (auto &buf : imf_storage)
            {
                buf.resize(n);
            }

            // Shared control variables
            double noise_amplitude = config_.noise_std * signal_std;
            int32_t global_valid_trials = 0;
            int32_t actual_imf_count = 0;
            bool stop_decomposition = false;

            // =================================================================
            // SINGLE PARALLEL REGION: Threads persist across all IMF stages
            // =================================================================
#pragma omp parallel
            {
                const int32_t tid = omp_get_thread_num();
                const int32_t n_threads = omp_get_num_threads();

                // ============================================================
                // ONE-TIME ALLOCATION PER THREAD (persists across all IMFs)
                // ============================================================
                LocalMeanComputer lm_computer(n, config_.boundary_extend);
                AlignedBuffer<double> thread_acc(n);
                AlignedBuffer<double> tl_perturbed(n);
                AlignedBuffer<double> tl_local_mean(n);

                // Manual loop distribution (static chunking)
                const int32_t chunk_size = (config_.ensemble_size + n_threads - 1) / n_threads;
                const int32_t my_start = tid * chunk_size;
                const int32_t my_end = std::min(my_start + chunk_size, config_.ensemble_size);

                // ============================================================
                // IMF EXTRACTION LOOP (threads stay alive)
                // ============================================================
                for (int32_t k = 0; k < config_.max_imfs; ++k)
                {
// Sync: All threads start this IMF together
#pragma omp barrier

                    // Check termination (set by master in previous iteration)
                    if (stop_decomposition)
                        break;

                    // Reset thread-local accumulator
                    thread_acc.zero();
                    int32_t thread_valid = 0;

                    // ========================================================
                    // ENSEMBLE LOOP (manual chunking, no omp for)
                    // ========================================================
                    for (int32_t i = my_start; i < my_end; ++i)
                    {
                        // Get k-th IMF of i-th noise realization
                        const double *noise_imf = noise_bank.get_noise_imf(i, k);

                        if (!noise_imf)
                        {
                            // This noise realization doesn't have k-th IMF
                            // Use zero noise (just current residue)
                            std::memcpy(tl_perturbed.data, r_current.data, n * sizeof(double));
                        }
                        else
                        {
                            // Perturb: r_current + ε_k * E_k(w^i)
                            // Note: r_current is read-only here, safe for concurrent read
                            const double *__restrict r = r_current.data;
                            const double *__restrict nz = noise_imf;
                            double *__restrict p = tl_perturbed.data;

                            EEMD_OMP_SIMD
                            for (int32_t j = 0; j < n; ++j)
                            {
                                p[j] = r[j] + noise_amplitude * nz[j];
                            }
                        }

                        // Compute local mean of perturbed signal
                        if (lm_computer.compute(tl_perturbed.data, n, tl_local_mean.data))
                        {
                            ++thread_valid;

                            // Accumulate local mean
                            double *__restrict acc = thread_acc.data;
                            const double *__restrict lm = tl_local_mean.data;

                            EEMD_OMP_SIMD
                            for (int32_t j = 0; j < n; ++j)
                            {
                                acc[j] += lm[j];
                            }
                        }
                    }

// ========================================================
// REDUCTION: Merge thread accumulators into global
// ========================================================
#pragma omp atomic
                    global_valid_trials += thread_valid;

#pragma omp critical
                    {
                        double *__restrict global = mean_accumulator.data;
                        const double *__restrict local = thread_acc.data;

                        EEMD_OMP_SIMD
                        for (int32_t j = 0; j < n; ++j)
                        {
                            global[j] += local[j];
                        }
                    }

// Sync: Wait for all threads to finish reduction
#pragma omp barrier

// ========================================================
// SINGLE THREAD: Process result, update residue
// ========================================================
#pragma omp single
                    {
                        if (global_valid_trials > 0)
                        {
                            // Average local means
                            const double scale = 1.0 / global_valid_trials;
                            double *__restrict acc = mean_accumulator.data;

                            EEMD_OMP_SIMD
                            for (int32_t j = 0; j < n; ++j)
                            {
                                acc[j] *= scale;
                            }

                            // Extract IMF: IMF_k = r_current - averaged_local_mean
                            const double *__restrict r = r_current.data;
                            const double *__restrict m = mean_accumulator.data;
                            double *__restrict out = imf_storage[actual_imf_count].data;

                            EEMD_OMP_SIMD
                            for (int32_t j = 0; j < n; ++j)
                            {
                                out[j] = r[j] - m[j];
                            }

                            ++actual_imf_count;

                            // Update residue: r_current = averaged_local_mean
                            std::memcpy(r_current.data, mean_accumulator.data, n * sizeof(double));

                            // Decay noise amplitude for next stage
                            noise_amplitude *= config_.noise_decay;

                            // Check stopping criteria for NEXT iteration
                            if (is_monotonic(r_current.data, n, config_.monotonic_threshold) ||
                                count_extrema(r_current.data, n) < config_.min_extrema)
                            {
                                stop_decomposition = true;
                            }
                        }
                        else
                        {
                            // No valid decompositions — stop
                            stop_decomposition = true;
                        }

                        // Reset for next IMF stage
                        global_valid_trials = 0;
                        mean_accumulator.zero();

                    } // implicit barrier at end of omp single

                } // end IMF loop

            } // end parallel region (threads join ONCE here)

            // Copy extracted IMFs to output
            imfs.resize(actual_imf_count);
            for (int32_t k = 0; k < actual_imf_count; ++k)
            {
                imfs[k].resize(n);
                std::memcpy(imfs[k].data(), imf_storage[k].data, n * sizeof(double));
            }

            // Final residue
            residue.resize(n);
            std::memcpy(residue.data(), r_current.data, n * sizeof(double));

            return true;
        }

        /**
         * Convenience wrapper that returns residue as last "IMF"
         */
        bool decompose_with_residue(
            const double *signal,
            int32_t n,
            std::vector<std::vector<double>> &imfs_and_residue)
        {
            std::vector<std::vector<double>> imfs;
            std::vector<double> residue;

            if (!decompose(signal, n, imfs, residue))
            {
                return false;
            }

            imfs_and_residue = std::move(imfs);
            imfs_and_residue.push_back(std::move(residue));

            return true;
        }

        ICEEMDANConfig &config() { return config_; }
        const ICEEMDANConfig &config() const { return config_; }

    private:
        ICEEMDANConfig config_;
    };

    // ============================================================================
    // Analysis Utilities - For Your Offline Market Analysis
    // ============================================================================

    /**
     * Compute Hurst exponent estimate using R/S analysis
     * H ≈ 0.5 → random walk (noise)
     * H < 0.5 → mean-reverting (structure)
     * H > 0.5 → trending (structure)
     */
    inline double estimate_hurst_rs(const double *signal, int32_t n)
    {
        if (n < 20)
            return 0.5;

        std::vector<double> log_n;
        std::vector<double> log_rs;

        // Try different window sizes
        for (int32_t win = 10; win <= n / 2; win = static_cast<int32_t>(win * 1.5))
        {
            int32_t n_windows = n / win;
            if (n_windows < 2)
                break;

            double rs_sum = 0.0;
            int32_t rs_count = 0;

            for (int32_t w = 0; w < n_windows; ++w)
            {
                const double *chunk = signal + w * win;

                // Mean
                double mean = 0.0;
                for (int32_t i = 0; i < win; ++i)
                {
                    mean += chunk[i];
                }
                mean /= win;

                // Cumulative deviations and range
                double cum = 0.0;
                double max_cum = -1e30;
                double min_cum = 1e30;
                double var = 0.0;

                for (int32_t i = 0; i < win; ++i)
                {
                    double dev = chunk[i] - mean;
                    cum += dev;
                    var += dev * dev;
                    max_cum = std::max(max_cum, cum);
                    min_cum = std::min(min_cum, cum);
                }

                double range = max_cum - min_cum;
                double std_dev = std::sqrt(var / win);

                if (std_dev > 1e-10)
                {
                    rs_sum += range / std_dev;
                    ++rs_count;
                }
            }

            if (rs_count > 0)
            {
                log_n.push_back(std::log(static_cast<double>(win)));
                log_rs.push_back(std::log(rs_sum / rs_count));
            }
        }

        if (log_n.size() < 2)
            return 0.5;

        // Linear regression: log(R/S) = H * log(n) + c
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
        int32_t m = static_cast<int32_t>(log_n.size());

        for (int32_t i = 0; i < m; ++i)
        {
            sum_x += log_n[i];
            sum_y += log_rs[i];
            sum_xy += log_n[i] * log_rs[i];
            sum_xx += log_n[i] * log_n[i];
        }

        double H = (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x);

        // Clamp to reasonable range
        return std::max(0.0, std::min(1.0, H));
    }

    /**
     * Compute spectral entropy (0 = pure tone, 1 = white noise)
     * Uses simple periodogram
     */
    inline double compute_spectral_entropy(const double *signal, int32_t n)
    {
        if (n < 4)
            return 1.0;

        // Allocate FFT buffers
        AlignedBuffer<MKL_Complex16> fft_in(n);
        AlignedBuffer<MKL_Complex16> fft_out(n);

        // Remove mean and copy to complex buffer
        double mean = 0.0;
        for (int32_t i = 0; i < n; ++i)
        {
            mean += signal[i];
        }
        mean /= n;

        for (int32_t i = 0; i < n; ++i)
        {
            fft_in[i].real = signal[i] - mean;
            fft_in[i].imag = 0.0;
        }

        // FFT
        DFTI_DESCRIPTOR_HANDLE desc = nullptr;
        MKL_LONG status = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, n);
        if (status != DFTI_NO_ERROR)
            return 1.0;

        DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiCommitDescriptor(desc);
        DftiComputeForward(desc, fft_in.data, fft_out.data);
        DftiFreeDescriptor(&desc);

        // Power spectral density (one-sided)
        int32_t n_freq = n / 2 + 1;
        std::vector<double> psd(n_freq);
        double total_power = 0.0;

        for (int32_t i = 0; i < n_freq; ++i)
        {
            double re = fft_out[i].real;
            double im = fft_out[i].imag;
            psd[i] = re * re + im * im;
            total_power += psd[i];
        }

        if (total_power < 1e-15)
            return 1.0;

        // Normalize to probability distribution and compute entropy
        double entropy = 0.0;
        for (int32_t i = 0; i < n_freq; ++i)
        {
            double p = psd[i] / total_power;
            if (p > 1e-15)
            {
                entropy -= p * std::log(p);
            }
        }

        // Normalize by max entropy (uniform distribution)
        double max_entropy = std::log(static_cast<double>(n_freq));
        return entropy / max_entropy;
    }

    /**
     * Compute sample entropy (complexity measure)
     * Lower = more regular/predictable, Higher = more random
     *
     * WARNING: O(N²) complexity! For large signals (N > 1000), consider:
     * - Downsampling the signal first
     * - Using a sliding window
     * - Skipping this metric for initial screening
     */
    inline double compute_sample_entropy(const double *signal, int32_t n,
                                         int32_t m = 2, double r_factor = 0.2)
    {
        if (n < m + 2)
            return 0.0;

        // Compute standard deviation
        double std_dev = compute_std(signal, n);
        double r = r_factor * std_dev;

        if (r < 1e-10)
            return 0.0;

        auto count_matches = [&](int32_t template_len) -> int64_t
        {
            int64_t count = 0;
            int32_t n_templates = n - template_len;

            for (int32_t i = 0; i < n_templates; ++i)
            {
                for (int32_t j = i + 1; j < n_templates; ++j)
                {
                    bool match = true;
                    for (int32_t k = 0; k < template_len && match; ++k)
                    {
                        if (std::abs(signal[i + k] - signal[j + k]) > r)
                        {
                            match = false;
                        }
                    }
                    if (match)
                        ++count;
                }
            }
            return count;
        };

        int64_t A = count_matches(m + 1); // matches of length m+1
        int64_t B = count_matches(m);     // matches of length m

        if (B == 0 || A == 0)
            return 0.0;

        return -std::log(static_cast<double>(A) / static_cast<double>(B));
    }

    /**
     * Analyze single IMF: returns struct with diagnostic metrics
     */
    struct IMFAnalysis
    {
        double hurst;
        double spectral_entropy;
        double sample_entropy;
        double energy;          // sum of squares
        double mean_frequency;  // weighted by power
        bool likely_noise;      // H ≈ 0.5 and high entropy
        bool likely_structure;  // H ≠ 0.5 or low entropy
    };

    inline IMFAnalysis analyze_imf(const double *imf, int32_t n, double sample_rate = 1.0)
    {
        IMFAnalysis result;

        result.hurst = estimate_hurst_rs(imf, n);
        result.spectral_entropy = compute_spectral_entropy(imf, n);
        result.sample_entropy = compute_sample_entropy(imf, n);

        // Energy
        result.energy = 0.0;
        for (int32_t i = 0; i < n; ++i)
        {
            result.energy += imf[i] * imf[i];
        }

        // Mean frequency via Hilbert
        std::vector<double> inst_freq(n);
        if (compute_instantaneous_frequency(imf, n, inst_freq.data(), sample_rate))
        {
            double freq_sum = 0.0;
            double weight_sum = 0.0;
            for (int32_t i = 0; i < n; ++i)
            {
                double w = imf[i] * imf[i]; // weight by amplitude squared
                freq_sum += std::abs(inst_freq[i]) * w;
                weight_sum += w;
            }
            result.mean_frequency = (weight_sum > 1e-10) ? freq_sum / weight_sum : 0.0;
        }
        else
        {
            result.mean_frequency = 0.0;
        }

        // Classification heuristics
        bool hurst_noise = (result.hurst > 0.4 && result.hurst < 0.6);
        bool high_entropy = (result.spectral_entropy > 0.8);

        result.likely_noise = hurst_noise && high_entropy;
        result.likely_structure = !hurst_noise || (result.spectral_entropy < 0.6);

        return result;
    }

} // namespace eemd

#endif // ICEEMDAN_MKL_HPP
