/**
 * EEMD-MKL Optimized for Intel Core i9-14900KF
 * 
 * Hardware Profile:
 *   - 8 P-cores (Golden Cove) @ 5.8GHz boost
 *   - 16 E-cores (Gracemont) @ 4.4GHz - IGNORED
 *   - 36MB L3 cache
 *   - AVX2 only (AVX-512 disabled in microcode)
 *   - DDR5-5600 dual channel
 * 
 * Optimization Strategy:
 *   - P-cores only, NO hyperthreading (8 threads)
 *   - Infinite blocktime (threads never sleep)
 *   - DAZ/FTZ enabled (no denormal traps)
 *   - MKL sequential for splines (avoid nested parallelism)
 *   - OpenMP for ensemble trials
 *   - AVX2 manual vectorization for hot loops
 * 
 * Usage:
 *   #include "eemd_mkl_14900kf.hpp"
 *   
 *   int main() {
 *       eemd::init_14900kf();  // Call ONCE at startup
 *       
 *       eemd::EEMDConfig config;
 *       config.ensemble_size = 100;
 *       
 *       eemd::EEMD_14900KF decomposer(config);
 *       // ... use decomposer ...
 *   }
 */

#ifndef EEMD_MKL_14900KF_HPP
#define EEMD_MKL_14900KF_HPP

#include <mkl.h>
#include <mkl_df.h>
#include <mkl_vsl.h>
#include <omp.h>

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>  // AVX2 intrinsics

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 14900KF Hardware Constants
// ============================================================================

#define EEMD_14900KF_P_CORES      8
#define EEMD_14900KF_E_CORES      16
#define EEMD_14900KF_L3_CACHE_KB  36864
#define EEMD_14900KF_CACHE_LINE   64

namespace eemd {

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
 * Initialize for 14900KF - call ONCE at program start
 * 
 * Configures:
 *   - 8 threads on P-cores only (no HT)
 *   - Infinite blocktime (no thread sleep)
 *   - DAZ/FTZ enabled
 *   - MKL sequential (OpenMP handles parallelism)
 */
inline void init_14900kf(bool verbose = false) {
    // 1. P-cores only, NO hyperthreading
    EEMD_SETENV("KMP_AFFINITY", "granularity=fine,compact,1,0");
    EEMD_SETENV("KMP_HW_SUBSET", "1s,8c,1t");  // 8 cores, 1 thread each
    
    // 2. Infinite blocktime - threads NEVER sleep
    EEMD_SETENV("KMP_BLOCKTIME", "infinite");
    EEMD_SETENV("KMP_LIBRARY", "turnaround");
    
    // 3. Force AVX2 (AVX-512 disabled on hybrid Intel)
    EEMD_SETENV("MKL_ENABLE_INSTRUCTIONS", "AVX2");
    
    // 4. Disable MKL dynamic threading
    mkl_set_dynamic(0);
    
    // 5. MKL sequential - we handle parallelism with OpenMP
    mkl_set_num_threads(1);
    
    // 6. OpenMP uses 8 P-cores
    omp_set_num_threads(EEMD_14900KF_P_CORES);
    
    // 7. DAZ/FTZ - flush denormals to zero (critical for latency!)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    
    if (verbose) {
        printf("╔═══════════════════════════════════════════════════════════╗\n");
        printf("║        EEMD-MKL: Intel Core i9-14900KF Mode               ║\n");
        printf("╠═══════════════════════════════════════════════════════════╣\n");
        printf("║  P-cores: 8 (E-cores ignored)                             ║\n");
        printf("║  Threads: 8 (1 per P-core, NO hyperthreading)             ║\n");
        printf("║  MKL: Sequential (spline ops)                             ║\n");
        printf("║  OpenMP: 8 threads (ensemble parallelism)                 ║\n");
        printf("║  Blocktime: INFINITE (threads never sleep)                ║\n");
        printf("║  DAZ/FTZ: ENABLED                                         ║\n");
        printf("║  Instructions: AVX2                                       ║\n");
        printf("╠═══════════════════════════════════════════════════════════╣\n");
        printf("║  WARNING: P-cores will spin at 100%% when idle.           ║\n");
        printf("╚═══════════════════════════════════════════════════════════╝\n\n");
    }
}

/**
 * Throughput mode for backtesting (uses HT, allows sleep)
 */
inline void init_14900kf_throughput(bool verbose = false) {
    EEMD_SETENV("KMP_AFFINITY", "granularity=fine,compact,1,0");
    EEMD_SETENV("KMP_HW_SUBSET", "1s,8c,2t");  // 8 cores, 2 threads each (HT)
    EEMD_SETENV("KMP_BLOCKTIME", "200");
    EEMD_SETENV("KMP_LIBRARY", "throughput");
    EEMD_SETENV("MKL_ENABLE_INSTRUCTIONS", "AVX2");
    
    mkl_set_dynamic(0);
    mkl_set_num_threads(1);  // Still keep MKL sequential
    omp_set_num_threads(16); // 8 P-cores * 2 HT
    
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    
    if (verbose) {
        printf("EEMD-MKL: 14900KF Throughput Mode (16 threads with HT)\n\n");
    }
}

// ============================================================================
// Memory Alignment (64-byte for cache line)
// ============================================================================

template<typename T>
class alignas(EEMD_14900KF_CACHE_LINE) AlignedBuffer {
public:
    T* data = nullptr;
    size_t size = 0;
    size_t capacity = 0;
    
    AlignedBuffer() = default;
    
    explicit AlignedBuffer(size_t n) {
        resize(n);
    }
    
    ~AlignedBuffer() {
        if (data) {
            mkl_free(data);
        }
    }
    
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data(other.data), size(other.size), capacity(other.capacity) {
        other.data = nullptr;
        other.size = other.capacity = 0;
    }
    
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            if (data) mkl_free(data);
            data = other.data;
            size = other.size;
            capacity = other.capacity;
            other.data = nullptr;
            other.size = other.capacity = 0;
        }
        return *this;
    }
    
    void resize(size_t n) {
        if (n > capacity) {
            if (data) mkl_free(data);
            capacity = n + (n / 4);  // 25% growth margin
            data = static_cast<T*>(mkl_malloc(capacity * sizeof(T), EEMD_14900KF_CACHE_LINE));
        }
        size = n;
    }
    
    void zero() {
        if (data && size > 0) {
            std::memset(data, 0, size * sizeof(T));
        }
    }
    
    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
};

// ============================================================================
// AVX2 Vectorized Operations
// ============================================================================

namespace avx2 {

/**
 * Vectorized signal + noise: dst[i] = signal[i] + noise[i]
 */
inline void add_signals(const double* signal, const double* noise, 
                        double* dst, int32_t n) {
    int32_t i = 0;
    
    // AVX2: 4 doubles per iteration
    for (; i + 4 <= n; i += 4) {
        __m256d s = _mm256_loadu_pd(signal + i);
        __m256d nz = _mm256_loadu_pd(noise + i);
        __m256d sum = _mm256_add_pd(s, nz);
        _mm256_storeu_pd(dst + i, sum);
    }
    
    // Scalar tail
    for (; i < n; ++i) {
        dst[i] = signal[i] + noise[i];
    }
}

/**
 * Vectorized accumulation: dst[i] += src[i]
 */
inline void accumulate(const double* src, double* dst, int32_t n) {
    int32_t i = 0;
    
    for (; i + 4 <= n; i += 4) {
        __m256d d = _mm256_loadu_pd(dst + i);
        __m256d s = _mm256_loadu_pd(src + i);
        __m256d sum = _mm256_add_pd(d, s);
        _mm256_storeu_pd(dst + i, sum);
    }
    
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

/**
 * Vectorized mean envelope and SD calculation
 * Returns: sd_num (sum of squared differences)
 */
inline double compute_mean_envelope_and_sd(
    const double* upper, const double* lower, const double* work,
    double* mean_env, int32_t n, double& sd_den
) {
    __m256d vsum_num = _mm256_setzero_pd();
    __m256d vsum_den = _mm256_setzero_pd();
    __m256d half = _mm256_set1_pd(0.5);
    
    int32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d u = _mm256_loadu_pd(upper + i);
        __m256d l = _mm256_loadu_pd(lower + i);
        __m256d w = _mm256_loadu_pd(work + i);
        
        // mean = 0.5 * (upper + lower)
        __m256d mean = _mm256_mul_pd(half, _mm256_add_pd(u, l));
        _mm256_storeu_pd(mean_env + i, mean);
        
        // diff = work - mean
        __m256d diff = _mm256_sub_pd(w, mean);
        
        // sd_num += diff * diff
        vsum_num = _mm256_fmadd_pd(diff, diff, vsum_num);
        
        // sd_den += work * work
        vsum_den = _mm256_fmadd_pd(w, w, vsum_den);
    }
    
    // Horizontal sum
    double sum_num[4], sum_den[4];
    _mm256_storeu_pd(sum_num, vsum_num);
    _mm256_storeu_pd(sum_den, vsum_den);
    
    double sd_num = sum_num[0] + sum_num[1] + sum_num[2] + sum_num[3];
    sd_den = sum_den[0] + sum_den[1] + sum_den[2] + sum_den[3];
    
    // Scalar tail
    for (; i < n; ++i) {
        mean_env[i] = 0.5 * (upper[i] + lower[i]);
        double diff = work[i] - mean_env[i];
        sd_num += diff * diff;
        sd_den += work[i] * work[i];
    }
    
    return sd_num;
}

/**
 * Vectorized subtraction: work[i] -= mean[i]
 */
inline void subtract_inplace(double* work, const double* mean, int32_t n) {
    int32_t i = 0;
    
    for (; i + 4 <= n; i += 4) {
        __m256d w = _mm256_loadu_pd(work + i);
        __m256d m = _mm256_loadu_pd(mean + i);
        __m256d diff = _mm256_sub_pd(w, m);
        _mm256_storeu_pd(work + i, diff);
    }
    
    for (; i < n; ++i) {
        work[i] -= mean[i];
    }
}

/**
 * Vectorized scale: dst[i] = src[i] * scale
 */
inline void scale(const double* src, double* dst, double scale_val, int32_t n) {
    __m256d vscale = _mm256_set1_pd(scale_val);
    int32_t i = 0;
    
    for (; i + 4 <= n; i += 4) {
        __m256d s = _mm256_loadu_pd(src + i);
        __m256d r = _mm256_mul_pd(s, vscale);
        _mm256_storeu_pd(dst + i, r);
    }
    
    for (; i < n; ++i) {
        dst[i] = src[i] * scale_val;
    }
}

/**
 * Vectorized signal statistics
 */
inline void compute_stats(const double* signal, int32_t n, 
                          double& mean, double& std_dev) {
    __m256d vsum = _mm256_setzero_pd();
    int32_t i = 0;
    
    // Pass 1: Mean
    for (; i + 4 <= n; i += 4) {
        __m256d s = _mm256_loadu_pd(signal + i);
        vsum = _mm256_add_pd(vsum, s);
    }
    
    double sums[4];
    _mm256_storeu_pd(sums, vsum);
    double total = sums[0] + sums[1] + sums[2] + sums[3];
    
    for (; i < n; ++i) {
        total += signal[i];
    }
    
    mean = total / n;
    
    // Pass 2: Variance
    __m256d vmean = _mm256_set1_pd(mean);
    __m256d vvar = _mm256_setzero_pd();
    
    for (i = 0; i + 4 <= n; i += 4) {
        __m256d s = _mm256_loadu_pd(signal + i);
        __m256d diff = _mm256_sub_pd(s, vmean);
        vvar = _mm256_fmadd_pd(diff, diff, vvar);
    }
    
    double vars[4];
    _mm256_storeu_pd(vars, vvar);
    double variance = vars[0] + vars[1] + vars[2] + vars[3];
    
    for (; i < n; ++i) {
        double diff = signal[i] - mean;
        variance += diff * diff;
    }
    
    std_dev = std::sqrt(variance / n);
}

}  // namespace avx2

// ============================================================================
// Configuration
// ============================================================================

struct EEMDConfig {
    int32_t max_imfs = 10;
    int32_t max_sift_iters = 50;      // Reduced for speed
    double sift_threshold = 0.05;
    int32_t ensemble_size = 100;
    double noise_std = 0.2;
    int32_t boundary_extend = 2;
    uint32_t rng_seed = 42;
};

// ============================================================================
// Thread Scratch Pad
// ============================================================================

struct ThreadScratchPad {
    std::vector<int32_t> max_idx;
    std::vector<int32_t> min_idx;
    std::vector<double> ext_x;
    std::vector<double> ext_y;
    
    explicit ThreadScratchPad(int32_t max_len) {
        max_idx.reserve(max_len / 2);
        min_idx.reserve(max_len / 2);
        ext_x.reserve(max_len + 20);
        ext_y.reserve(max_len + 20);
    }
};

// ============================================================================
// Peak Finding (Branchless)
// ============================================================================

inline void find_maxima_noalloc(const double* signal, int32_t n, 
                                std::vector<int32_t>& out) {
    out.clear();
    for (int32_t i = 1; i < n - 1; ++i) {
        int is_max = (signal[i] > signal[i-1]) & (signal[i] > signal[i+1]);
        if (is_max) out.push_back(i);
    }
}

inline void find_minima_noalloc(const double* signal, int32_t n,
                                std::vector<int32_t>& out) {
    out.clear();
    for (int32_t i = 1; i < n - 1; ++i) {
        int is_min = (signal[i] < signal[i-1]) & (signal[i] < signal[i+1]);
        if (is_min) out.push_back(i);
    }
}

inline int32_t count_zero_crossings(const double* signal, int32_t n) {
    int32_t count = 0;
    for (int32_t i = 1; i < n; ++i) {
        count += (signal[i-1] * signal[i] < 0.0) ? 1 : 0;
    }
    return count;
}

// ============================================================================
// Boundary Extension
// ============================================================================

inline void extend_extrema_noalloc(
    const std::vector<int32_t>& indices,
    const double* signal,
    int32_t signal_len,
    int32_t extend_count,
    std::vector<double>& out_x,
    std::vector<double>& out_y,
    int32_t& out_count,
    int32_t& original_start
) {
    out_x.clear();
    out_y.clear();
    
    const int32_t n_orig = static_cast<int32_t>(indices.size());
    
    if (n_orig < 2) {
        out_count = n_orig;
        original_start = 0;
        for (int32_t i = 0; i < n_orig; ++i) {
            out_x.push_back(static_cast<double>(indices[i]));
            out_y.push_back(signal[indices[i]]);
        }
        return;
    }
    
    const int32_t left_extend = std::min(extend_count, n_orig - 1);
    const int32_t right_extend = std::min(extend_count, n_orig - 1);
    
    // Check coverage
    double leftmost_x = static_cast<double>(indices[0]);
    double rightmost_x = static_cast<double>(indices[n_orig - 1]);
    
    if (left_extend > 0) {
        leftmost_x = 2.0 * indices[0] - indices[left_extend];
    }
    if (right_extend > 0) {
        rightmost_x = 2.0 * indices[n_orig-1] - indices[n_orig-1-right_extend];
    }
    
    const bool need_left = (leftmost_x > 0.0);
    const bool need_right = (rightmost_x < static_cast<double>(signal_len - 1));
    
    // Left boundary
    if (need_left) {
        double x0 = static_cast<double>(indices[0]);
        double x1 = static_cast<double>(indices[1]);
        double y0 = signal[indices[0]];
        double y1 = signal[indices[1]];
        double slope = (y1 - y0) / (x1 - x0);
        out_x.push_back(-1.0);
        out_y.push_back(y0 + slope * (-1.0 - x0));
    }
    
    // Mirror left
    for (int32_t i = 0; i < left_extend; ++i) {
        const int32_t src_idx = left_extend - i;
        out_x.push_back(2.0 * indices[0] - indices[src_idx]);
        out_y.push_back(signal[indices[src_idx]]);
    }
    
    original_start = static_cast<int32_t>(out_x.size());
    
    // Original
    for (int32_t i = 0; i < n_orig; ++i) {
        out_x.push_back(static_cast<double>(indices[i]));
        out_y.push_back(signal[indices[i]]);
    }
    
    // Mirror right
    for (int32_t i = 0; i < right_extend; ++i) {
        const int32_t src_idx = n_orig - 2 - i;
        out_x.push_back(2.0 * indices[n_orig-1] - indices[src_idx]);
        out_y.push_back(signal[indices[src_idx]]);
    }
    
    // Right boundary
    if (need_right) {
        double x0 = static_cast<double>(indices[n_orig - 2]);
        double x1 = static_cast<double>(indices[n_orig - 1]);
        double y0 = signal[indices[n_orig - 2]];
        double y1 = signal[indices[n_orig - 1]];
        double slope = (y1 - y0) / (x1 - x0);
        out_x.push_back(static_cast<double>(signal_len));
        out_y.push_back(y1 + slope * (signal_len - x1));
    }
    
    out_count = static_cast<int32_t>(out_x.size());
}

// ============================================================================
// MKL Spline Wrapper
// ============================================================================

class MKLSpline {
public:
    MKLSpline() = default;
    
    ~MKLSpline() {
        cleanup();
    }
    
    MKLSpline(const MKLSpline&) = delete;
    MKLSpline& operator=(const MKLSpline&) = delete;
    
    bool construct(const double* x, const double* y, int32_t n) {
        cleanup();
        
        if (n < 2) return false;
        
        const MKL_INT mkl_n = static_cast<MKL_INT>(n);
        const MKL_INT n_coeffs = 4 * (mkl_n - 1);
        
        if (coeffs_.capacity < static_cast<size_t>(n_coeffs)) {
            coeffs_.resize(n_coeffs);
        }
        coeffs_.size = n_coeffs;
        
        MKL_INT status = dfdNewTask1D(&task_, mkl_n, x, 
                                       DF_NON_UNIFORM_PARTITION, 1, y, DF_NO_HINT);
        if (status != DF_STATUS_OK) return false;
        
        status = dfdEditPPSpline1D(task_, DF_PP_CUBIC, DF_PP_NATURAL,
                                    DF_BC_FREE_END, nullptr, DF_NO_IC, nullptr,
                                    coeffs_.data, DF_NO_HINT);
        if (status != DF_STATUS_OK) { cleanup(); return false; }
        
        status = dfdConstruct1D(task_, DF_PP_SPLINE, DF_METHOD_STD);
        if (status != DF_STATUS_OK) { cleanup(); return false; }
        
        valid_ = true;
        return true;
    }
    
    bool evaluate(const double* sites, double* results, int32_t n_sites) const {
        if (!valid_) return false;
        
        const MKL_INT dorder[] = {1};
        MKL_INT status = dfdInterpolate1D(task_, DF_INTERP, DF_METHOD_PP,
                                           static_cast<MKL_INT>(n_sites), sites,
                                           DF_SORTED_DATA, 1, dorder, nullptr,
                                           results, DF_NO_HINT, nullptr);
        return (status == DF_STATUS_OK);
    }
    
private:
    void cleanup() {
        if (task_) {
            dfDeleteTask(&task_);
            task_ = nullptr;
        }
        valid_ = false;
    }
    
    DFTaskPtr task_ = nullptr;
    AlignedBuffer<double> coeffs_;
    bool valid_ = false;
};

// ============================================================================
// Sifter with AVX2
// ============================================================================

class Sifter_14900KF {
public:
    explicit Sifter_14900KF(int32_t max_len, const EEMDConfig& cfg)
        : config_(cfg)
        , scratch_(max_len)
        , work_(max_len)
        , upper_env_(max_len)
        , lower_env_(max_len)
        , mean_env_(max_len)
        , sites_(max_len)
    {
        for (int32_t i = 0; i < max_len; ++i) {
            sites_[i] = static_cast<double>(i);
        }
    }
    
    bool sift_imf(double* signal, double* imf, int32_t n) {
        std::memcpy(work_.data, signal, n * sizeof(double));
        
        for (int32_t iter = 0; iter < config_.max_sift_iters; ++iter) {
            find_maxima_noalloc(work_.data, n, scratch_.max_idx);
            find_minima_noalloc(work_.data, n, scratch_.min_idx);
            
            if (scratch_.max_idx.size() < 2 || scratch_.min_idx.size() < 2) {
                return false;
            }
            
            // Upper envelope
            int32_t ext_count, ext_start;
            extend_extrema_noalloc(scratch_.max_idx, work_.data, n,
                                   config_.boundary_extend, scratch_.ext_x,
                                   scratch_.ext_y, ext_count, ext_start);
            
            if (!upper_spline_.construct(scratch_.ext_x.data(), 
                                          scratch_.ext_y.data(), ext_count)) {
                return false;
            }
            if (!upper_spline_.evaluate(sites_.data, upper_env_.data, n)) {
                return false;
            }
            
            // Lower envelope
            extend_extrema_noalloc(scratch_.min_idx, work_.data, n,
                                   config_.boundary_extend, scratch_.ext_x,
                                   scratch_.ext_y, ext_count, ext_start);
            
            if (!lower_spline_.construct(scratch_.ext_x.data(),
                                          scratch_.ext_y.data(), ext_count)) {
                return false;
            }
            if (!lower_spline_.evaluate(sites_.data, lower_env_.data, n)) {
                return false;
            }
            
            // AVX2: Compute mean envelope and SD
            double sd_den = 0.0;
            double sd_num = avx2::compute_mean_envelope_and_sd(
                upper_env_.data, lower_env_.data, work_.data,
                mean_env_.data, n, sd_den
            );
            
            // AVX2: Subtract mean
            avx2::subtract_inplace(work_.data, mean_env_.data, n);
            
            const double sd = (sd_den > 1e-15) ? sd_num / sd_den : 0.0;
            
            if (sd < config_.sift_threshold) break;
            
            const int32_t n_ext = static_cast<int32_t>(
                scratch_.max_idx.size() + scratch_.min_idx.size());
            const int32_t n_zero = count_zero_crossings(work_.data, n);
            
            if (std::abs(n_ext - n_zero) <= 1 && sd < config_.sift_threshold * 10) {
                break;
            }
        }
        
        std::memcpy(imf, work_.data, n * sizeof(double));
        avx2::subtract_inplace(signal, work_.data, n);
        
        return true;
    }
    
private:
    const EEMDConfig& config_;
    ThreadScratchPad scratch_;
    AlignedBuffer<double> work_;
    AlignedBuffer<double> upper_env_;
    AlignedBuffer<double> lower_env_;
    AlignedBuffer<double> mean_env_;
    AlignedBuffer<double> sites_;
    MKLSpline upper_spline_;
    MKLSpline lower_spline_;
};

// ============================================================================
// EEMD Main Class - Optimized for 14900KF (8 P-cores)
// ============================================================================

class EEMD_14900KF {
public:
    explicit EEMD_14900KF(const EEMDConfig& config = EEMDConfig())
        : config_(config)
    {}
    
    bool decompose(
        const double* signal,
        int32_t n,
        std::vector<std::vector<double>>& imfs,
        int32_t& n_imfs
    ) {
        if (n < 4) return false;
        
        // AVX2: Compute signal stats
        double signal_mean, signal_std;
        avx2::compute_stats(signal, n, signal_mean, signal_std);
        const double noise_amplitude = config_.noise_std * signal_std;
        
        const int32_t max_imfs = config_.max_imfs;
        
        // Global accumulator
        std::vector<AlignedBuffer<double>> global_sum(max_imfs);
        for (auto& buf : global_sum) {
            buf.resize(n);
            buf.zero();
        }
        
        int32_t global_max_imfs = 0;
        
        // Parallel region - 8 P-cores
        #pragma omp parallel
        {
            // Thread-local buffers
            std::vector<AlignedBuffer<double>> thread_sum(max_imfs);
            for (auto& buf : thread_sum) {
                buf.resize(n);
                buf.zero();
            }
            
            std::vector<AlignedBuffer<double>> local_imfs(max_imfs);
            for (auto& buf : local_imfs) {
                buf.resize(n);
            }
            
            AlignedBuffer<double> noisy_signal(n);
            AlignedBuffer<double> noise(n);
            
            int32_t thread_max_imfs = 0;
            
            // VSL RNG per thread
            VSLStreamStatePtr stream;
            vslNewStream(&stream, VSL_BRNG_MT2203 + omp_get_thread_num(),
                         config_.rng_seed + omp_get_thread_num());
            
            Sifter_14900KF sifter(n, config_);
            
            #pragma omp for schedule(dynamic)
            for (int32_t trial = 0; trial < config_.ensemble_size; ++trial) {
                // Generate noise
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream,
                              n, noise.data, 0.0, noise_amplitude);
                
                // AVX2: Add noise
                avx2::add_signals(signal, noise.data, noisy_signal.data, n);
                
                // EMD
                int32_t imf_count = 0;
                for (int32_t k = 0; k < max_imfs; ++k) {
                    if (!sifter.sift_imf(noisy_signal.data, local_imfs[k].data, n)) {
                        break;
                    }
                    ++imf_count;
                }
                
                thread_max_imfs = std::max(thread_max_imfs, imf_count);
                
                // AVX2: Thread-local accumulation
                for (int32_t k = 0; k < imf_count; ++k) {
                    avx2::accumulate(local_imfs[k].data, thread_sum[k].data, n);
                }
            }
            
            vslDeleteStream(&stream);
            
            // Single critical section per thread
            #pragma omp critical
            {
                global_max_imfs = std::max(global_max_imfs, thread_max_imfs);
                
                for (int32_t k = 0; k < max_imfs; ++k) {
                    avx2::accumulate(thread_sum[k].data, global_sum[k].data, n);
                }
            }
        }
        
        n_imfs = global_max_imfs;
        
        // AVX2: Compute ensemble average
        const double scale = 1.0 / config_.ensemble_size;
        imfs.resize(n_imfs);
        
        for (int32_t k = 0; k < n_imfs; ++k) {
            imfs[k].resize(n);
            avx2::scale(global_sum[k].data, imfs[k].data(), scale, n);
        }
        
        return true;
    }
    
    bool decompose_emd(
        const double* signal,
        int32_t n,
        std::vector<std::vector<double>>& imfs,
        std::vector<double>& residue
    ) {
        if (n < 4) return false;
        
        imfs.clear();
        residue.resize(n);
        std::memcpy(residue.data(), signal, n * sizeof(double));
        
        Sifter_14900KF sifter(n, config_);
        std::vector<double> imf(n);
        
        for (int32_t k = 0; k < config_.max_imfs; ++k) {
            if (!sifter.sift_imf(residue.data(), imf.data(), n)) {
                break;
            }
            imfs.push_back(imf);
        }
        
        return !imfs.empty();
    }
    
private:
    EEMDConfig config_;
};

// ============================================================================
// Convenience Aliases
// ============================================================================

using EEMD = EEMD_14900KF;
using Sifter = Sifter_14900KF;

}  // namespace eemd

#endif  // EEMD_MKL_14900KF_HPP
