/**
 * EEMD (Ensemble Empirical Mode Decomposition) with Intel MKL
 *
 * Optimized Implementation:
 * - DF_UNIFORM_PARTITION for O(1) knot lookup (vs O(log K) binary search)
 * - Raw pointer hot paths (no std::vector overhead in inner loops)
 * - Fused mean/SD/update loop (single memory pass)
 * - Thread-local accumulation (one critical section per thread)
 * - Zero-malloc scratch pads (pre-allocated, reused buffers)
 * - Capacity-aware MKL buffers (grow-only)
 *
 * Dependencies:
 * - MKL Data Fitting (df) for cubic spline interpolation
 * - MKL VSL for Gaussian noise generation
 * - OpenMP for ensemble parallelization
 *
 * License: MIT
 */

#ifndef EEMD_MKL_HPP
#define EEMD_MKL_HPP

#include <mkl.h>
#include <mkl_df.h>
#include <mkl_vsl.h>
#include <omp.h>

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Compiler-aware SIMD pragma
#if defined(__INTEL_LLVM_COMPILER) || defined(__GNUC__) || defined(__clang__)
#define EEMD_OMP_SIMD _Pragma("omp simd")
#define EEMD_OMP_SIMD_REDUCTION(op, var) _Pragma("omp simd reduction(" #op ":" #var ")")
#define EEMD_OMP_SIMD_REDUCTION2(op, v1, v2) _Pragma("omp simd reduction(" #op ":" #v1 "," #v2 ")")
#else
#define EEMD_OMP_SIMD
#define EEMD_OMP_SIMD_REDUCTION(op, var)
#define EEMD_OMP_SIMD_REDUCTION2(op, v1, v2)
#endif

// ============================================================================
// Hardware Constants (tune for your CPU)
// ============================================================================

#define EEMD_P_CORES 8
#define EEMD_CACHE_LINE 64

// ============================================================================
// Platform Configuration
// ============================================================================

#ifdef _WIN32
#include <intrin.h>
#define EEMD_SETENV(name, value) _putenv_s(name, value)
#else
#define EEMD_SETENV(name, value) setenv(name, value, 1)
#endif

/**
 * Low-latency mode: P-cores only, threads never sleep
 */
inline void eemd_init_low_latency(int n_cores = EEMD_P_CORES, bool verbose = false)
{
    EEMD_SETENV("KMP_AFFINITY", "granularity=fine,compact,1,0");

    char subset[32];
    snprintf(subset, sizeof(subset), "1s,%dc,1t", n_cores);
    EEMD_SETENV("KMP_HW_SUBSET", subset);

    EEMD_SETENV("KMP_BLOCKTIME", "infinite");
    EEMD_SETENV("KMP_LIBRARY", "turnaround");
    EEMD_SETENV("MKL_ENABLE_INSTRUCTIONS", "AVX2");

    mkl_set_dynamic(0);
    mkl_set_num_threads(1);
    omp_set_num_threads(n_cores);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    if (verbose)
    {
        printf("EEMD-MKL: Low-latency mode (%d cores, infinite blocktime, DAZ/FTZ)\n", n_cores);
    }
}

/**
 * Throughput mode: uses hyperthreading, allows thread sleep
 */
inline void eemd_init_throughput(int n_cores = EEMD_P_CORES, bool verbose = false)
{
    EEMD_SETENV("KMP_AFFINITY", "granularity=fine,compact,1,0");

    char subset[32];
    snprintf(subset, sizeof(subset), "1s,%dc,2t", n_cores);
    EEMD_SETENV("KMP_HW_SUBSET", subset);

    EEMD_SETENV("KMP_BLOCKTIME", "200");
    EEMD_SETENV("KMP_LIBRARY", "throughput");
    EEMD_SETENV("MKL_ENABLE_INSTRUCTIONS", "AVX2");

    mkl_set_dynamic(0);
    mkl_set_num_threads(1);
    omp_set_num_threads(n_cores * 2);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    if (verbose)
    {
        printf("EEMD-MKL: Throughput mode (%d threads with HT)\n", n_cores * 2);
    }
}

namespace eemd
{

    // ============================================================================
    // Configuration
    // ============================================================================

    struct EEMDConfig
    {
        int32_t max_imfs = 10;
        int32_t max_sift_iters = 100;
        double sift_threshold = 0.05;
        int32_t ensemble_size = 100;
        double noise_std = 0.2;
        int32_t boundary_extend = 2;
        uint32_t rng_seed = 42;
    };

    // ============================================================================
    // Aligned Buffer - Grow-Only, 64-byte Aligned
    // ============================================================================

    template <typename T>
    struct AlignedBuffer
    {
        T *data = nullptr;
        size_t size = 0;
        size_t capacity = 0;

        AlignedBuffer() = default;

        explicit AlignedBuffer(size_t n)
        {
            if (n > 0)
            {
                data = static_cast<T *>(mkl_malloc(n * sizeof(T), EEMD_CACHE_LINE));
                if (!data)
                    throw std::bad_alloc();
                size = n;
                capacity = n;
            }
        }

        ~AlignedBuffer()
        {
            if (data)
                mkl_free(data);
        }

        AlignedBuffer(const AlignedBuffer &) = delete;
        AlignedBuffer &operator=(const AlignedBuffer &) = delete;

        AlignedBuffer(AlignedBuffer &&other) noexcept
            : data(other.data), size(other.size), capacity(other.capacity)
        {
            other.data = nullptr;
            other.size = 0;
            other.capacity = 0;
        }

        AlignedBuffer &operator=(AlignedBuffer &&other) noexcept
        {
            if (this != &other)
            {
                if (data)
                    mkl_free(data);
                data = other.data;
                size = other.size;
                capacity = other.capacity;
                other.data = nullptr;
                other.size = 0;
                other.capacity = 0;
            }
            return *this;
        }

        void resize(size_t n)
        {
            if (n > capacity)
            {
                if (data)
                    mkl_free(data);
                data = (n > 0) ? static_cast<T *>(mkl_malloc(n * sizeof(T), EEMD_CACHE_LINE)) : nullptr;
                if (n > 0 && !data)
                    throw std::bad_alloc();
                capacity = n;
            }
            size = n;
        }

        void zero()
        {
            if (data && size > 0)
            {
                std::memset(data, 0, size * sizeof(T));
            }
        }

        T &operator[](size_t i) { return data[i]; }
        const T &operator[](size_t i) const { return data[i]; }
    };

    // ============================================================================
    // Thread Scratch Pad - Fixed-Size Arrays with Counters
    // ============================================================================

    struct ThreadScratchPad
    {
        // Fixed-size arrays (resized once at construction)
        std::vector<int32_t> max_idx;
        std::vector<int32_t> min_idx;
        std::vector<double> ext_x;
        std::vector<double> ext_y;

        // Counters (avoids vector size() calls in hot path)
        int32_t n_max = 0;
        int32_t n_min = 0;
        int32_t n_ext = 0;
        int32_t ext_start = 0;

        explicit ThreadScratchPad(int32_t n)
        {
            const int32_t max_extrema = n / 2 + 2;
            const int32_t ext_size = n + 20;

            max_idx.resize(max_extrema);
            min_idx.resize(max_extrema);
            ext_x.resize(ext_size);
            ext_y.resize(ext_size);
        }
    };

    // ============================================================================
    // Peak Finding - Raw Pointer Output (No vector push_back)
    // ============================================================================

    inline void find_maxima_raw(
        const double *__restrict signal,
        int32_t n,
        int32_t *__restrict out_idx,
        int32_t &out_count)
    {
        int32_t count = 0;

        if (n >= 3)
        {
            for (int32_t i = 1; i < n - 1; ++i)
            {
                const bool peak = (signal[i] > signal[i - 1]) & (signal[i] > signal[i + 1]);
                if (peak)
                {
                    out_idx[count++] = i;
                }
            }
        }

        out_count = count;
    }

    inline void find_minima_raw(
        const double *__restrict signal,
        int32_t n,
        int32_t *__restrict out_idx,
        int32_t &out_count)
    {
        int32_t count = 0;

        if (n >= 3)
        {
            for (int32_t i = 1; i < n - 1; ++i)
            {
                const bool trough = (signal[i] < signal[i - 1]) & (signal[i] < signal[i + 1]);
                if (trough)
                {
                    out_idx[count++] = i;
                }
            }
        }

        out_count = count;
    }

    inline int32_t count_zero_crossings(const double *signal, int32_t n)
    {
        int32_t count = 0;
        for (int32_t i = 1; i < n; ++i)
        {
            count += ((signal[i - 1] >= 0.0) != (signal[i] >= 0.0));
        }
        return count;
    }

    // ============================================================================
    // Boundary Extension - Raw Pointer Output
    // ============================================================================

    inline void extend_extrema_raw(
        const int32_t *__restrict indices,
        int32_t n_indices,
        const double *__restrict signal,
        int32_t signal_len,
        int32_t extend_count,
        double *__restrict out_x,
        double *__restrict out_y,
        int32_t &out_count,
        int32_t &original_start)
    {
        if (n_indices < 2)
        {
            out_count = n_indices;
            original_start = 0;
            for (int32_t i = 0; i < n_indices; ++i)
            {
                out_x[i] = static_cast<double>(indices[i]);
                out_y[i] = signal[indices[i]];
            }
            return;
        }

        const int32_t left_ext = std::min(extend_count, n_indices - 1);
        const int32_t right_ext = std::min(extend_count, n_indices - 1);

        // Compute coverage after mirroring
        double leftmost_x = static_cast<double>(indices[0]);
        double rightmost_x = static_cast<double>(indices[n_indices - 1]);

        if (left_ext > 0)
        {
            leftmost_x = 2.0 * indices[0] - indices[left_ext];
        }
        if (right_ext > 0)
        {
            rightmost_x = 2.0 * indices[n_indices - 1] - indices[n_indices - 1 - right_ext];
        }

        const bool need_left = (leftmost_x > 0.0);
        const bool need_right = (rightmost_x < static_cast<double>(signal_len - 1));

        int32_t pos = 0;

        // Left boundary point (linear extrapolation)
        if (need_left)
        {
            const double x0 = static_cast<double>(indices[0]);
            const double x1 = static_cast<double>(indices[1]);
            const double y0 = signal[indices[0]];
            const double y1 = signal[indices[1]];
            const double slope = (y1 - y0) / (x1 - x0);
            out_x[pos] = -1.0;
            out_y[pos] = y0 + slope * (-1.0 - x0);
            ++pos;
        }

        // Mirror left
        for (int32_t i = 0; i < left_ext; ++i)
        {
            const int32_t src = left_ext - i;
            out_x[pos] = 2.0 * indices[0] - indices[src];
            out_y[pos] = signal[indices[src]];
            ++pos;
        }

        original_start = pos;

        // Original extrema
        for (int32_t i = 0; i < n_indices; ++i)
        {
            out_x[pos] = static_cast<double>(indices[i]);
            out_y[pos] = signal[indices[i]];
            ++pos;
        }

        // Mirror right
        for (int32_t i = 0; i < right_ext; ++i)
        {
            const int32_t src = n_indices - 2 - i;
            out_x[pos] = 2.0 * indices[n_indices - 1] - indices[src];
            out_y[pos] = signal[indices[src]];
            ++pos;
        }

        // Right boundary point
        if (need_right)
        {
            const double x0 = static_cast<double>(indices[n_indices - 2]);
            const double x1 = static_cast<double>(indices[n_indices - 1]);
            const double y0 = signal[indices[n_indices - 2]];
            const double y1 = signal[indices[n_indices - 1]];
            const double slope = (y1 - y0) / (x1 - x0);
            out_x[pos] = static_cast<double>(signal_len);
            out_y[pos] = y1 + slope * (signal_len - x1);
            ++pos;
        }

        out_count = pos;
    }

    // ============================================================================
    // MKL Spline - With DF_UNIFORM_PARTITION Optimization
    // ============================================================================

    class MKLSpline
    {
    public:
        MKLSpline() = default;
        ~MKLSpline() { cleanup(); }

        MKLSpline(const MKLSpline &) = delete;
        MKLSpline &operator=(const MKLSpline &) = delete;

        bool construct(const double *x, const double *y, int32_t n)
        {
            cleanup();

            if (n < 2)
                return false;

            const MKL_INT mkl_n = static_cast<MKL_INT>(n);
            const MKL_INT required = 4 * (mkl_n - 1);

            if (coeffs_.capacity < static_cast<size_t>(required))
            {
                coeffs_.resize(required);
            }
            coeffs_.size = required;

            MKL_INT status = dfdNewTask1D(&task_, mkl_n, x,
                                          DF_NON_UNIFORM_PARTITION, 1, y, DF_NO_HINT);
            if (status != DF_STATUS_OK)
                return false;

            task_valid_ = true;

            status = dfdEditPPSpline1D(task_, DF_PP_CUBIC, DF_PP_NATURAL,
                                       DF_BC_FREE_END, nullptr, DF_NO_IC, nullptr,
                                       coeffs_.data, DF_NO_HINT);
            if (status != DF_STATUS_OK)
            {
                cleanup();
                return false;
            }

            status = dfdConstruct1D(task_, DF_PP_SPLINE, DF_METHOD_STD);
            if (status != DF_STATUS_OK)
            {
                cleanup();
                return false;
            }

            spline_valid_ = true;
            return true;
        }

        /**
         * Evaluate at uniform grid [0, 1, 2, ..., n_sites-1]
         * Uses DF_UNIFORM_PARTITION for O(1) knot lookup instead of O(log K)
         */
        bool evaluate_uniform(double *results, int32_t n_sites) const
        {
            if (!spline_valid_)
                return false;

            const double interval[2] = {0.0, static_cast<double>(n_sites - 1)};
            const MKL_INT mkl_n = static_cast<MKL_INT>(n_sites);
            const MKL_INT dorder[] = {1};

            MKL_INT status = dfdInterpolate1D(
                task_,
                DF_INTERP,
                DF_METHOD_PP,
                mkl_n,
                interval,
                DF_UNIFORM_PARTITION, // O(1) knot lookup
                1,
                dorder,
                nullptr,
                results,
                DF_NO_HINT,
                nullptr);

            return (status == DF_STATUS_OK);
        }

        /**
         * Evaluate at arbitrary sorted sites (fallback)
         */
        bool evaluate(const double *sites, double *results, int32_t n_sites) const
        {
            if (!spline_valid_)
                return false;

            const MKL_INT mkl_n = static_cast<MKL_INT>(n_sites);
            const MKL_INT dorder[] = {1};

            MKL_INT status = dfdInterpolate1D(
                task_,
                DF_INTERP,
                DF_METHOD_PP,
                mkl_n,
                sites,
                DF_SORTED_DATA,
                1,
                dorder,
                nullptr,
                results,
                DF_NO_HINT,
                nullptr);

            return (status == DF_STATUS_OK);
        }

    private:
        void cleanup()
        {
            if (task_valid_)
            {
                dfDeleteTask(&task_);
                task_valid_ = false;
            }
            spline_valid_ = false;
        }

        DFTaskPtr task_ = nullptr;
        bool task_valid_ = false;
        bool spline_valid_ = false;
        AlignedBuffer<double> coeffs_;
    };

    // ============================================================================
    // Sifter - Fused Loops, Raw Pointers
    // ============================================================================

    class Sifter
    {
    public:
        explicit Sifter(int32_t max_len, const EEMDConfig &cfg)
            : config_(cfg), max_len_(max_len), scratch_(max_len), work_(max_len), upper_env_(max_len), lower_env_(max_len)
        {
        }

        bool sift_imf(double *signal, double *imf, int32_t n)
        {
            std::memcpy(work_.data, signal, n * sizeof(double));

            // Raw pointers for hot path
            int32_t *p_max = scratch_.max_idx.data();
            int32_t *p_min = scratch_.min_idx.data();
            double *p_ext_x = scratch_.ext_x.data();
            double *p_ext_y = scratch_.ext_y.data();

            for (int32_t iter = 0; iter < config_.max_sift_iters; ++iter)
            {
                // Find extrema (raw pointer output)
                find_maxima_raw(work_.data, n, p_max, scratch_.n_max);
                find_minima_raw(work_.data, n, p_min, scratch_.n_min);

                if (scratch_.n_max < 2 || scratch_.n_min < 2)
                {
                    return false;
                }

                // Upper envelope
                extend_extrema_raw(p_max, scratch_.n_max, work_.data, n,
                                   config_.boundary_extend, p_ext_x, p_ext_y,
                                   scratch_.n_ext, scratch_.ext_start);

                if (!upper_spline_.construct(p_ext_x, p_ext_y, scratch_.n_ext))
                {
                    return false;
                }
                if (!upper_spline_.evaluate_uniform(upper_env_.data, n))
                {
                    return false;
                }

                // Lower envelope (reuse ext buffers)
                extend_extrema_raw(p_min, scratch_.n_min, work_.data, n,
                                   config_.boundary_extend, p_ext_x, p_ext_y,
                                   scratch_.n_ext, scratch_.ext_start);

                if (!lower_spline_.construct(p_ext_x, p_ext_y, scratch_.n_ext))
                {
                    return false;
                }
                if (!lower_spline_.evaluate_uniform(lower_env_.data, n))
                {
                    return false;
                }

                // Fused: compute mean, update work, calculate SD in single pass
                double sd_num = 0.0;
                double sd_den = 0.0;

                const double *__restrict upper = upper_env_.data;
                const double *__restrict lower = lower_env_.data;
                double *__restrict w = work_.data;

                EEMD_OMP_SIMD_REDUCTION2(+, sd_num, sd_den)
                for (int32_t i = 0; i < n; ++i)
                {
                    const double u = upper[i];
                    const double l = lower[i];
                    const double val = w[i];
                    const double mean = 0.5 * (u + l);
                    const double diff = val - mean;

                    sd_num += mean * mean; // sum of mean^2
                    sd_den += val * val;   // sum of original^2

                    w[i] = diff; // update in-place
                }

                const double sd = (sd_den > 1e-15) ? sd_num / sd_den : 0.0;

                if (sd < config_.sift_threshold)
                    break;

                // Early termination: extrema ≈ zero crossings
                const int32_t n_extrema = scratch_.n_max + scratch_.n_min;
                const int32_t n_zero = count_zero_crossings(work_.data, n);

                if (std::abs(n_extrema - n_zero) <= 1 && sd < config_.sift_threshold * 10)
                {
                    break;
                }
            }

            std::memcpy(imf, work_.data, n * sizeof(double));

            // Subtract IMF from signal
            double *__restrict sig = signal;
            const double *__restrict w = work_.data;

            EEMD_OMP_SIMD
            for (int32_t i = 0; i < n; ++i)
            {
                sig[i] -= w[i];
            }

            return true;
        }

    private:
        const EEMDConfig &config_;
        int32_t max_len_;

        ThreadScratchPad scratch_;
        AlignedBuffer<double> work_;
        AlignedBuffer<double> upper_env_;
        AlignedBuffer<double> lower_env_;

        MKLSpline upper_spline_;
        MKLSpline lower_spline_;
    };

    // ============================================================================
    // EEMD Main Class
    // ============================================================================

    class EEMD
    {
    public:
        explicit EEMD(const EEMDConfig &config = EEMDConfig())
            : config_(config)
        {
        }

        bool decompose(
            const double *signal,
            int32_t n,
            std::vector<std::vector<double>> &imfs,
            int32_t &n_imfs)
        {
            if (n < 4)
                return false;

            // Compute signal stats
            double signal_mean = 0.0;
            EEMD_OMP_SIMD_REDUCTION(+, signal_mean)
            for (int32_t i = 0; i < n; ++i)
            {
                signal_mean += signal[i];
            }
            signal_mean /= n;

            double signal_var = 0.0;
            EEMD_OMP_SIMD_REDUCTION(+, signal_var)
            for (int32_t i = 0; i < n; ++i)
            {
                const double d = signal[i] - signal_mean;
                signal_var += d * d;
            }
            const double signal_std = std::sqrt(signal_var / n);
            const double noise_amplitude = config_.noise_std * signal_std;

            const int32_t max_imfs = config_.max_imfs;

            // Global accumulator
            std::vector<AlignedBuffer<double>> global_sum(max_imfs);
            for (auto &buf : global_sum)
            {
                buf.resize(n);
                buf.zero();
            }

            int32_t global_max_imfs = 0;

#pragma omp parallel
            {
                const int32_t tid = omp_get_thread_num();

                // Thread-local accumulator
                std::vector<AlignedBuffer<double>> thread_sum(max_imfs);
                for (auto &buf : thread_sum)
                {
                    buf.resize(n);
                    buf.zero();
                }

                int32_t thread_max_imfs = 0;

                // Thread-local RNG
                VSLStreamStatePtr stream = nullptr;
                vslNewStream(&stream, VSL_BRNG_MT19937, config_.rng_seed + tid * 1000);

                // Thread-local buffers
                AlignedBuffer<double> noisy_signal(n);
                AlignedBuffer<double> noise(n);
                std::vector<AlignedBuffer<double>> local_imfs(max_imfs);
                for (auto &buf : local_imfs)
                {
                    buf.resize(n);
                }

                Sifter sifter(n, config_);

#pragma omp for schedule(dynamic)
                for (int32_t trial = 0; trial < config_.ensemble_size; ++trial)
                {
                    // Generate noise
                    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream,
                                  n, noise.data, 0.0, noise_amplitude);

                    // Add noise
                    const double *__restrict sig = signal;
                    const double *__restrict nz = noise.data;
                    double *__restrict ns = noisy_signal.data;

                    EEMD_OMP_SIMD
                    for (int32_t i = 0; i < n; ++i)
                    {
                        ns[i] = sig[i] + nz[i];
                    }

                    // EMD
                    int32_t imf_count = 0;
                    for (int32_t k = 0; k < max_imfs; ++k)
                    {
                        if (!sifter.sift_imf(noisy_signal.data, local_imfs[k].data, n))
                        {
                            break;
                        }
                        ++imf_count;
                    }

                    thread_max_imfs = std::max(thread_max_imfs, imf_count);

                    // Thread-local accumulation
                    for (int32_t k = 0; k < imf_count; ++k)
                    {
                        double *__restrict ts = thread_sum[k].data;
                        const double *__restrict li = local_imfs[k].data;

                        EEMD_OMP_SIMD
                        for (int32_t i = 0; i < n; ++i)
                        {
                            ts[i] += li[i];
                        }
                    }
                }

                vslDeleteStream(&stream);

// Single critical section per thread
#pragma omp critical
                {
                    global_max_imfs = std::max(global_max_imfs, thread_max_imfs);

                    for (int32_t k = 0; k < max_imfs; ++k)
                    {
                        double *__restrict gs = global_sum[k].data;
                        const double *__restrict ts = thread_sum[k].data;

                        EEMD_OMP_SIMD
                        for (int32_t i = 0; i < n; ++i)
                        {
                            gs[i] += ts[i];
                        }
                    }
                }
            }

            n_imfs = global_max_imfs;

            // Ensemble average
            const double scale = 1.0 / config_.ensemble_size;
            imfs.resize(n_imfs);

            for (int32_t k = 0; k < n_imfs; ++k)
            {
                imfs[k].resize(n);
                const double *__restrict gs = global_sum[k].data;
                double *__restrict out = imfs[k].data();

                EEMD_OMP_SIMD
                for (int32_t i = 0; i < n; ++i)
                {
                    out[i] = gs[i] * scale;
                }
            }

            return true;
        }

        bool decompose_emd(
            const double *signal,
            int32_t n,
            std::vector<std::vector<double>> &imfs,
            std::vector<double> &residue)
        {
            if (n < 4)
                return false;

            AlignedBuffer<double> work(n);
            std::memcpy(work.data, signal, n * sizeof(double));

            Sifter sifter(n, config_);

            imfs.clear();
            imfs.reserve(config_.max_imfs);

            for (int32_t k = 0; k < config_.max_imfs; ++k)
            {
                std::vector<double> imf(n);
                if (!sifter.sift_imf(work.data, imf.data(), n))
                {
                    break;
                }
                imfs.push_back(std::move(imf));
            }

            residue.resize(n);
            std::memcpy(residue.data(), work.data, n * sizeof(double));

            return true;
        }

        EEMDConfig &config() { return config_; }
        const EEMDConfig &config() const { return config_; }

    private:
        EEMDConfig config_;
    };

    // ============================================================================
    // Hilbert Transform - Instantaneous Frequency
    // ============================================================================

    inline bool compute_instantaneous_frequency(
        const double *imf,
        int32_t n,
        double *inst_freq,
        double sample_rate = 1.0)
    {
        AlignedBuffer<MKL_Complex16> fft_in(n);
        AlignedBuffer<MKL_Complex16> fft_out(n);
        AlignedBuffer<MKL_Complex16> analytic(n);

        for (int32_t i = 0; i < n; ++i)
        {
            fft_in[i].real = imf[i];
            fft_in[i].imag = 0.0;
        }

        DFTI_DESCRIPTOR_HANDLE desc = nullptr;
        MKL_LONG status = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, n);
        if (status != DFTI_NO_ERROR)
            return false;

        DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        status = DftiCommitDescriptor(desc);
        if (status != DFTI_NO_ERROR)
        {
            DftiFreeDescriptor(&desc);
            return false;
        }

        status = DftiComputeForward(desc, fft_in.data, fft_out.data);
        if (status != DFTI_NO_ERROR)
        {
            DftiFreeDescriptor(&desc);
            return false;
        }

        // Hilbert transform
        analytic[0] = fft_out[0];
        const int32_t half = n / 2;

        for (int32_t i = 1; i < half; ++i)
        {
            analytic[i].real = 2.0 * fft_out[i].real;
            analytic[i].imag = 2.0 * fft_out[i].imag;
        }

        if (n % 2 == 0)
        {
            analytic[half] = fft_out[half];
        }

        for (int32_t i = half + 1; i < n; ++i)
        {
            analytic[i].real = 0.0;
            analytic[i].imag = 0.0;
        }

        status = DftiComputeBackward(desc, analytic.data, fft_in.data);
        DftiFreeDescriptor(&desc);
        if (status != DFTI_NO_ERROR)
            return false;

        const double scale = 1.0 / n;
        const double freq_scale = sample_rate / (2.0 * M_PI);

        for (int32_t i = 1; i < n - 1; ++i)
        {
            const double ar = fft_in[i].real * scale;
            const double ai = fft_in[i].imag * scale;
            const double ar_next = fft_in[i + 1].real * scale;
            const double ai_next = fft_in[i + 1].imag * scale;

            double dphase = std::atan2(ai_next, ar_next) - std::atan2(ai, ar);

            while (dphase > M_PI)
                dphase -= 2.0 * M_PI;
            while (dphase < -M_PI)
                dphase += 2.0 * M_PI;

            inst_freq[i] = dphase * freq_scale;
        }

        inst_freq[0] = inst_freq[1];
        inst_freq[n - 1] = inst_freq[n - 2];

        return true;
    }

} // namespace eemd

#endif // EEMD_MKL_HPP