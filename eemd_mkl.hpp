/**
 * EEMD (Ensemble Empirical Mode Decomposition) with Intel MKL
 *
 * Optimized Implementation:
 * - Thread-local accumulation (no critical section per trial)
 * - Zero-malloc scratch pads (pre-allocated, reused buffers)
 * - Capacity-aware MKL buffers (no shrink, grow-only)
 * - Branchless peak finding (pipeline-friendly)
 *
 * Dependencies:
 * - MKL Data Fitting (df) for cubic spline interpolation
 * - MKL VSL for Gaussian noise generation
 * - OpenMP for ensemble parallelization
 *
 * Author: Generated for TUGBARS
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
#include <iostream> // For MKL error diagnostics

// Windows compatibility: M_PI is not standard C++
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Compiler-aware SIMD pragma
// MSVC standard OpenMP doesn't support #pragma omp simd
// Intel ICX, GCC, and Clang do
#if defined(__INTEL_LLVM_COMPILER) || defined(__GNUC__) || defined(__clang__)
#define EEMD_OMP_SIMD _Pragma("omp simd")
#define EEMD_OMP_SIMD_REDUCTION(op, var) _Pragma("omp simd reduction(" #op ":" #var ")")
#define EEMD_OMP_SIMD_REDUCTION2(op, v1, v2) _Pragma("omp simd reduction(" #op ":" #v1 "," #v2 ")")
#else
// MSVC or unknown compiler - disable SIMD pragmas
#define EEMD_OMP_SIMD
#define EEMD_OMP_SIMD_REDUCTION(op, var)
#define EEMD_OMP_SIMD_REDUCTION2(op, v1, v2)
#endif

// Define EEMD_DEBUG to enable error output, otherwise silent
#ifndef EEMD_DEBUG
#define EEMD_SILENT
#endif

namespace eemd
{

    // ============================================================================
    // Configuration Constants
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
    // Memory Management - Capacity-Aware Aligned Buffer
    // ============================================================================

    template <typename T>
    struct AlignedBuffer
    {
        T *data = nullptr;
        size_t size = 0;
        size_t capacity = 0; // Track capacity separately

        AlignedBuffer() = default;

        explicit AlignedBuffer(size_t n)
        {
            if (n > 0)
            {
                data = static_cast<T *>(mkl_malloc(n * sizeof(T), 64));
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

        // Grow-only resize: never shrinks allocation
        void resize(size_t n)
        {
            if (n > capacity)
            {
                // Need more space - reallocate
                if (data)
                    mkl_free(data);
                data = (n > 0) ? static_cast<T *>(mkl_malloc(n * sizeof(T), 64)) : nullptr;
                if (n > 0 && !data)
                    throw std::bad_alloc();
                capacity = n;
            }
            size = n;
        }

        // Reserve without changing size
        void reserve(size_t n)
        {
            if (n > capacity)
            {
                T *new_data = static_cast<T *>(mkl_malloc(n * sizeof(T), 64));
                if (!new_data)
                    throw std::bad_alloc();
                if (data)
                {
                    std::memcpy(new_data, data, size * sizeof(T));
                    mkl_free(data);
                }
                data = new_data;
                capacity = n;
            }
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
    // Thread Scratch Pad - Zero-Malloc Strategy
    // ============================================================================

    struct ThreadScratchPad
    {
        std::vector<int32_t> max_idx;
        std::vector<int32_t> min_idx;
        std::vector<double> ext_x;
        std::vector<double> ext_y;

        // Pre-allocate worst-case size
        explicit ThreadScratchPad(int32_t n)
        {
            const int32_t max_extrema = n / 2 + 1;
            const int32_t ext_size = n + 20; // Room for boundary extension

            max_idx.reserve(max_extrema);
            min_idx.reserve(max_extrema);
            ext_x.reserve(ext_size);
            ext_y.reserve(ext_size);
        }

        void clear()
        {
            max_idx.clear();
            min_idx.clear();
            ext_x.clear();
            ext_y.clear();
        }
    };

    // ============================================================================
    // Peak Finding - Branchless / Pipeline-Friendly
    // ============================================================================

    /**
     * Find local maxima - branchless inner loop, no allocation
     * Writes indices to pre-allocated output vector
     */
    inline void find_maxima_noalloc(
        const double *signal,
        int32_t n,
        std::vector<int32_t> &out)
    {
        out.clear(); // Does not deallocate, just sets size=0

        if (n < 3)
            return;

        for (int32_t i = 1; i < n - 1; ++i)
        {
            // Branchless comparison - avoids pipeline stalls on noisy data
            const bool left_rise = signal[i] > signal[i - 1];
            const bool right_fall = signal[i] > signal[i + 1];

            if (left_rise & right_fall)
            {
                out.push_back(i); // No alloc - we reserved capacity
            }
        }

        // Handle flat tops with a second pass if needed
        // (Flat tops are rare in noisy EEMD context, so branch is predictable)
    }

    /**
     * Find local maxima with flat-top handling - no allocation version
     */
    inline void find_maxima_flat_noalloc(
        const double *signal,
        int32_t n,
        std::vector<int32_t> &out)
    {
        out.clear();

        if (n < 3)
            return;

        int32_t i = 1;
        while (i < n - 1)
        {
            if (signal[i] > signal[i - 1])
            {
                int32_t plateau_start = i;
                // Skip flat plateau
                while (i < n - 1 && signal[i] == signal[i + 1])
                {
                    ++i;
                }
                if (i < n - 1 && signal[i] > signal[i + 1])
                {
                    out.push_back((plateau_start + i) / 2);
                }
            }
            ++i;
        }
    }

    /**
     * Find local minima - branchless, no allocation
     */
    inline void find_minima_noalloc(
        const double *signal,
        int32_t n,
        std::vector<int32_t> &out)
    {
        out.clear();

        if (n < 3)
            return;

        for (int32_t i = 1; i < n - 1; ++i)
        {
            const bool left_fall = signal[i] < signal[i - 1];
            const bool right_rise = signal[i] < signal[i + 1];

            if (left_fall & right_rise)
            {
                out.push_back(i);
            }
        }
    }

    /**
     * Find local minima with flat-bottom handling - no allocation
     */
    inline void find_minima_flat_noalloc(
        const double *signal,
        int32_t n,
        std::vector<int32_t> &out)
    {
        out.clear();

        if (n < 3)
            return;

        int32_t i = 1;
        while (i < n - 1)
        {
            if (signal[i] < signal[i - 1])
            {
                int32_t plateau_start = i;
                while (i < n - 1 && signal[i] == signal[i + 1])
                {
                    ++i;
                }
                if (i < n - 1 && signal[i] < signal[i + 1])
                {
                    out.push_back((plateau_start + i) / 2);
                }
            }
            ++i;
        }
    }

    /**
     * Count zero crossings
     */
    inline int32_t count_zero_crossings(const double *signal, int32_t n)
    {
        int32_t count = 0;
        for (int32_t i = 1; i < n; ++i)
        {
            // Branchless: xor of sign bits
            const bool sign_prev = signal[i - 1] >= 0.0;
            const bool sign_curr = signal[i] >= 0.0;
            count += (sign_prev != sign_curr);
        }
        return count;
    }

    // ============================================================================
    // Boundary Extension - Guaranteed Coverage of [0, n-1]
    // ============================================================================

    /**
     * Extend extrema to guarantee the spline covers the full signal range [0, n-1]
     * Uses mirror boundary conditions plus explicit edge points if needed
     */
    inline void extend_extrema_noalloc(
        const std::vector<int32_t> &indices,
        const double *signal,
        int32_t signal_len,
        int32_t extend_count,
        std::vector<double> &out_x,
        std::vector<double> &out_y,
        int32_t &out_count,
        int32_t &original_start)
    {
        out_x.clear();
        out_y.clear();

        const int32_t n_orig = static_cast<int32_t>(indices.size());

        if (n_orig < 2)
        {
            out_count = n_orig;
            original_start = 0;
            for (int32_t i = 0; i < n_orig; ++i)
            {
                out_x.push_back(static_cast<double>(indices[i]));
                out_y.push_back(signal[indices[i]]);
            }
            return;
        }

        const int32_t left_extend = std::min(extend_count, n_orig - 1);
        const int32_t right_extend = std::min(extend_count, n_orig - 1);

        // First, compute what the leftmost and rightmost x will be after mirror extension
        double leftmost_x = static_cast<double>(indices[0]);
        double rightmost_x = static_cast<double>(indices[n_orig - 1]);

        if (left_extend > 0)
        {
            // Leftmost mirrored point
            const double x_first = static_cast<double>(indices[0]);
            const double x_mirror_src = static_cast<double>(indices[left_extend]);
            leftmost_x = 2.0 * x_first - x_mirror_src;
        }

        if (right_extend > 0)
        {
            // Rightmost mirrored point
            const double x_last = static_cast<double>(indices[n_orig - 1]);
            const double x_mirror_src = static_cast<double>(indices[n_orig - 1 - right_extend]);
            rightmost_x = 2.0 * x_last - x_mirror_src;
        }

        // Check if we need to add explicit boundary points
        const bool need_left_boundary = (leftmost_x > 0.0);
        const bool need_right_boundary = (rightmost_x < static_cast<double>(signal_len - 1));

        // Add left boundary point if needed (extrapolate to x = -1 or use x = 0)
        if (need_left_boundary)
        {
            // Add a point at x = -1 (or further left) to ensure coverage of x = 0
            double boundary_x = -1.0;
            // Linear extrapolation from first two extrema
            double x0 = static_cast<double>(indices[0]);
            double x1 = static_cast<double>(indices[1]);
            double y0 = signal[indices[0]];
            double y1 = signal[indices[1]];
            double slope = (y1 - y0) / (x1 - x0);
            double boundary_y = y0 + slope * (boundary_x - x0);
            out_x.push_back(boundary_x);
            out_y.push_back(boundary_y);
        }

        // Mirror left boundary
        for (int32_t i = 0; i < left_extend; ++i)
        {
            const int32_t src_idx = left_extend - i;
            const double x_orig = static_cast<double>(indices[src_idx]);
            const double x_first = static_cast<double>(indices[0]);

            out_x.push_back(2.0 * x_first - x_orig);
            out_y.push_back(signal[indices[src_idx]]);
        }

        original_start = static_cast<int32_t>(out_x.size());

        // Original extrema
        for (int32_t i = 0; i < n_orig; ++i)
        {
            out_x.push_back(static_cast<double>(indices[i]));
            out_y.push_back(signal[indices[i]]);
        }

        // Mirror right boundary
        for (int32_t i = 0; i < right_extend; ++i)
        {
            const int32_t src_idx = n_orig - 2 - i;
            const double x_orig = static_cast<double>(indices[src_idx]);
            const double x_last = static_cast<double>(indices[n_orig - 1]);

            out_x.push_back(2.0 * x_last - x_orig);
            out_y.push_back(signal[indices[src_idx]]);
        }

        // Add right boundary point if needed
        if (need_right_boundary)
        {
            // Add a point at x = signal_len (beyond the last index)
            double boundary_x = static_cast<double>(signal_len);
            // Linear extrapolation from last two extrema
            double x0 = static_cast<double>(indices[n_orig - 2]);
            double x1 = static_cast<double>(indices[n_orig - 1]);
            double y0 = signal[indices[n_orig - 2]];
            double y1 = signal[indices[n_orig - 1]];
            double slope = (y1 - y0) / (x1 - x0);
            double boundary_y = y1 + slope * (boundary_x - x1);
            out_x.push_back(boundary_x);
            out_y.push_back(boundary_y);
        }

        out_count = static_cast<int32_t>(out_x.size());
    }

    // ============================================================================
    // MKL Spline - Capacity-Aware Buffer Reuse
    // ============================================================================

    class MKLSpline
    {
    public:
        MKLSpline() = default;
        ~MKLSpline() { cleanup_task(); }

        MKLSpline(const MKLSpline &) = delete;
        MKLSpline &operator=(const MKLSpline &) = delete;

        /**
         * Construct spline - reuses coefficient buffer if large enough
         */
        bool construct(const double *x, const double *y, int32_t n)
        {
            cleanup_task();

            if (n < 2)
                return false;

            n_points_ = n;

            // Cast to MKL_INT to ensure correct integer size
            const MKL_INT mkl_n = static_cast<MKL_INT>(n);

            // Grow-only: only reallocate if we need more space
            const MKL_INT required_coeffs = 4 * (mkl_n - 1);
            if (coeffs_.capacity < static_cast<size_t>(required_coeffs))
            {
                coeffs_.resize(required_coeffs);
            }
            coeffs_.size = required_coeffs;

            // Create Data Fitting task
            MKL_INT status = dfdNewTask1D(
                &task_,
                mkl_n,
                x,
                DF_NON_UNIFORM_PARTITION,
                1,
                y,
                DF_NO_HINT);

            // DEBUG: Check for MKL errors (especially LP64/ILP64 mismatch)
            if (status != DF_STATUS_OK)
            {
#ifdef EEMD_DEBUG
                std::cerr << "[MKL ERROR] dfdNewTask1D failed, status=" << status
                          << ", n=" << n << ", mkl_n=" << mkl_n
                          << ", sizeof(MKL_INT)=" << sizeof(MKL_INT) << "\n";
                if (status == -3)
                {
                    std::cerr << "  -> Likely LP64/ILP64 mismatch! Link with mkl_intel_lp64, not ilp64.\n";
                }
#endif
                return false;
            }
            task_valid_ = true;

            const MKL_INT spline_order = DF_PP_CUBIC;
            const MKL_INT spline_type = DF_PP_NATURAL;

            status = dfdEditPPSpline1D(
                task_,
                spline_order,
                spline_type,
                DF_BC_FREE_END,
                nullptr,
                DF_NO_IC,
                nullptr,
                coeffs_.data,
                DF_NO_HINT);

            if (status != DF_STATUS_OK)
            {
#ifdef EEMD_DEBUG
                std::cerr << "[MKL ERROR] dfdEditPPSpline1D failed, status=" << status << "\n";
#endif
                cleanup_task();
                return false;
            }

            status = dfdConstruct1D(task_, DF_PP_SPLINE, DF_METHOD_STD);

            if (status != DF_STATUS_OK)
            {
#ifdef EEMD_DEBUG
                std::cerr << "[MKL ERROR] dfdConstruct1D failed, status=" << status << "\n";
#endif
                cleanup_task();
                return false;
            }

            spline_valid_ = true;
            return true;
        }

        bool evaluate(const double *sites, double *results, int32_t n_sites) const
        {
            if (!spline_valid_)
                return false;

            const MKL_INT mkl_n_sites = static_cast<MKL_INT>(n_sites);

            MKL_INT status = dfdInterpolate1D(
                task_,
                DF_INTERP,
                DF_METHOD_PP,
                mkl_n_sites,
                sites,
                DF_NON_UNIFORM_PARTITION,
                1,
                nullptr,
                nullptr,
                results,
                DF_NO_HINT,
                nullptr);

            if (status != DF_STATUS_OK)
            {
#ifdef EEMD_DEBUG
                std::cerr << "[MKL ERROR] dfdInterpolate1D failed, status=" << status << "\n";
#endif
                return false;
            }

            return true;
        }

        bool is_valid() const { return spline_valid_; }

    private:
        void cleanup_task()
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
        int32_t n_points_ = 0;
        AlignedBuffer<double> coeffs_; // Reused across calls
    };

    // ============================================================================
    // Sifting Process - Uses Scratch Pad
    // ============================================================================

    class Sifter
    {
    public:
        explicit Sifter(int32_t max_len, const EEMDConfig &cfg)
            : config_(cfg), max_len_(max_len), scratch_(max_len), work_(max_len), upper_env_(max_len), lower_env_(max_len), mean_env_(max_len), sites_(max_len)
        {
            for (int32_t i = 0; i < max_len; ++i)
            {
                sites_[i] = static_cast<double>(i);
            }
        }

        /**
         * Extract one IMF using sifting
         * @return true if IMF extracted, false if residue
         */
        bool sift_imf(double *signal, double *imf, int32_t n)
        {
            std::memcpy(work_.data, signal, n * sizeof(double));

            for (int32_t iter = 0; iter < config_.max_sift_iters; ++iter)
            {
                // Find extrema using scratch buffers - NO ALLOCATION
                find_maxima_noalloc(work_.data, n, scratch_.max_idx);
                find_minima_noalloc(work_.data, n, scratch_.min_idx);

                if (scratch_.max_idx.size() < 2 || scratch_.min_idx.size() < 2)
                {
#ifdef EEMD_DEBUG
                    std::cerr << "[EEMD DEBUG] Not enough extrema: maxima="
                              << scratch_.max_idx.size() << ", minima="
                              << scratch_.min_idx.size() << ", n=" << n << "\n";
#endif
                    return false; // Residue
                }

                // CRITICAL: MKL Data Fitting keeps pointers to input arrays!
                // We must construct AND evaluate each spline before reusing buffers.

                // Upper envelope (maxima)
                int32_t max_count, max_start;
                extend_extrema_noalloc(
                    scratch_.max_idx, work_.data, n, config_.boundary_extend,
                    scratch_.ext_x, scratch_.ext_y, max_count, max_start);

                if (!upper_spline_.construct(
                        scratch_.ext_x.data(), scratch_.ext_y.data(), max_count))
                {
                    return false;
                }

                // Evaluate IMMEDIATELY before overwriting buffers
                if (!upper_spline_.evaluate(sites_.data, upper_env_.data, n))
                {
                    return false;
                }

                // Lower envelope (minima) - now safe to reuse ext_x/ext_y
                int32_t min_count, min_start;
                extend_extrema_noalloc(
                    scratch_.min_idx, work_.data, n, config_.boundary_extend,
                    scratch_.ext_x, scratch_.ext_y, min_count, min_start);

                if (!lower_spline_.construct(
                        scratch_.ext_x.data(), scratch_.ext_y.data(), min_count))
                {
                    return false;
                }

                if (!lower_spline_.evaluate(sites_.data, lower_env_.data, n))
                {
                    return false;
                }

                // Compute mean and SD - vectorized
                double sd_num = 0.0;
                double sd_den = 0.0;

                EEMD_OMP_SIMD_REDUCTION2(+, sd_num, sd_den)
                for (int32_t i = 0; i < n; ++i)
                {
                    mean_env_[i] = 0.5 * (upper_env_[i] + lower_env_[i]);
                    const double diff = work_[i] - mean_env_[i];
                    sd_num += diff * diff;
                    sd_den += work_[i] * work_[i];
                }

                // Update working signal
                EEMD_OMP_SIMD
                for (int32_t i = 0; i < n; ++i)
                {
                    work_[i] -= mean_env_[i];
                }

                const double sd = (sd_den > 1e-15) ? sd_num / sd_den : 0.0;

                if (sd < config_.sift_threshold)
                {
                    break;
                }

                const int32_t n_extrema = static_cast<int32_t>(
                    scratch_.max_idx.size() + scratch_.min_idx.size());
                const int32_t n_zero = count_zero_crossings(work_.data, n);

                if (std::abs(n_extrema - n_zero) <= 1 && sd < config_.sift_threshold * 10)
                {
                    break;
                }
            }

            std::memcpy(imf, work_.data, n * sizeof(double));

            EEMD_OMP_SIMD
            for (int32_t i = 0; i < n; ++i)
            {
                signal[i] -= work_[i];
            }

            return true;
        }

    private:
        const EEMDConfig &config_;
        int32_t max_len_;

        ThreadScratchPad scratch_; // Reusable scratch memory
        AlignedBuffer<double> work_;
        AlignedBuffer<double> upper_env_;
        AlignedBuffer<double> lower_env_;
        AlignedBuffer<double> mean_env_;
        AlignedBuffer<double> sites_;

        MKLSpline upper_spline_;
        MKLSpline lower_spline_;
    };

    // ============================================================================
    // EEMD Main Class - Thread-Local Accumulation
    // ============================================================================

    class EEMD
    {
    public:
        explicit EEMD(const EEMDConfig &config = EEMDConfig())
            : config_(config)
        {
        }

        /**
         * Perform EEMD decomposition with optimized parallel accumulation
         */
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

            // OPTIMIZATION 1: Global accumulator
            std::vector<AlignedBuffer<double>> global_sum(max_imfs);
            for (auto &buf : global_sum)
            {
                buf.resize(n);
                buf.zero();
            }

            // Track max IMFs across all trials
            int32_t global_max_imfs = 0;

#pragma omp parallel
            {
                const int32_t thread_id = omp_get_thread_num();

                // OPTIMIZATION 2: Thread-local accumulator (avoids critical per trial)
                std::vector<AlignedBuffer<double>> thread_sum(max_imfs);
                for (auto &buf : thread_sum)
                {
                    buf.resize(n);
                    buf.zero();
                }

                int32_t thread_max_imfs = 0;

                // Thread-local VSL stream
                VSLStreamStatePtr stream = nullptr;
                const uint32_t thread_seed = config_.rng_seed +
                                             static_cast<uint32_t>(thread_id) * 1000;
                vslNewStream(&stream, VSL_BRNG_MT19937, thread_seed);

                // Thread-local buffers (allocated once per thread)
                AlignedBuffer<double> noisy_signal(n);
                AlignedBuffer<double> noise(n);
                std::vector<AlignedBuffer<double>> local_imfs(max_imfs);
                for (auto &buf : local_imfs)
                {
                    buf.resize(n);
                }

                // Thread-local sifter (contains its own scratch pad)
                Sifter sifter(n, config_);

#pragma omp for schedule(dynamic)
                for (int32_t trial = 0; trial < config_.ensemble_size; ++trial)
                {
                    // Generate noise
                    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream,
                                  n, noise.data, 0.0, noise_amplitude);

                    // Add noise to signal
                    EEMD_OMP_SIMD
                    for (int32_t i = 0; i < n; ++i)
                    {
                        noisy_signal[i] = signal[i] + noise[i];
                    }

                    // EMD decomposition
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

                    // OPTIMIZATION 3: Accumulate to thread-local buffer (NO CRITICAL!)
                    for (int32_t k = 0; k < imf_count; ++k)
                    {
                        EEMD_OMP_SIMD
                        for (int32_t i = 0; i < n; ++i)
                        {
                            thread_sum[k][i] += local_imfs[k][i];
                        }
                    }
                }

                vslDeleteStream(&stream);

// OPTIMIZATION 4: Single critical section per thread at the end
#pragma omp critical
                {
                    global_max_imfs = std::max(global_max_imfs, thread_max_imfs);

                    for (int32_t k = 0; k < max_imfs; ++k)
                    {
                        EEMD_OMP_SIMD
                        for (int32_t i = 0; i < n; ++i)
                        {
                            global_sum[k][i] += thread_sum[k][i];
                        }
                    }
                }
            }

            n_imfs = global_max_imfs;

            // Compute ensemble average
            const double scale = 1.0 / config_.ensemble_size;
            imfs.resize(n_imfs);

            for (int32_t k = 0; k < n_imfs; ++k)
            {
                imfs[k].resize(n);
                EEMD_OMP_SIMD
                for (int32_t i = 0; i < n; ++i)
                {
                    imfs[k][i] = global_sum[k][i] * scale;
                }
            }

            return true;
        }

        /**
         * Simple EMD (no ensemble) - useful for testing
         */
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
    // Utility: Instantaneous Frequency via Hilbert Transform
    // ============================================================================

    inline bool compute_instantaneous_frequency(
        const double *imf,
        int32_t n,
        double *inst_freq,
        double sample_rate = 1.0)
    {
        const MKL_LONG fft_n = n;

        AlignedBuffer<MKL_Complex16> fft_in(n);
        AlignedBuffer<MKL_Complex16> fft_out(n);
        AlignedBuffer<MKL_Complex16> analytic(n);

        EEMD_OMP_SIMD
        for (int32_t i = 0; i < n; ++i)
        {
            fft_in[i].real = imf[i];
            fft_in[i].imag = 0.0;
        }

        DFTI_DESCRIPTOR_HANDLE desc = nullptr;
        MKL_LONG status = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, fft_n);
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

        // Hilbert transform in frequency domain
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

            const double phase = std::atan2(ai, ar);
            const double phase_next = std::atan2(ai_next, ar_next);
            double dphase = phase_next - phase;

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