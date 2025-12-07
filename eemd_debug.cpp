/**
 * EEMD Debug Test - Diagnose why IMFs = 0
 */

// Enable debug output for this file
#define EEMD_DEBUG

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <mkl.h>
#include <mkl_df.h>
#include <iostream>
#include <iomanip>
#include <vector>

// Check MKL integer size
void check_mkl_config()
{
    std::cout << "=== MKL Configuration ===\n";
    std::cout << "sizeof(MKL_INT): " << sizeof(MKL_INT) << " bytes\n";
    std::cout << "sizeof(int32_t): " << sizeof(int32_t) << " bytes\n";
    std::cout << "sizeof(int):     " << sizeof(int) << " bytes\n";

    if (sizeof(MKL_INT) != sizeof(int32_t))
    {
        std::cout << "WARNING: MKL_INT != int32_t! You may have ILP64 linkage!\n";
    }

    MKLVersion ver;
    mkl_get_version(&ver);
    std::cout << "MKL Version: " << ver.MajorVersion << "."
              << ver.MinorVersion << "." << ver.UpdateVersion << "\n";
    std::cout << "MKL Platform: " << ver.Platform << "\n\n";
}

// Test raw MKL spline API
bool test_mkl_spline_raw()
{
    std::cout << "=== Raw MKL Spline Test ===\n";

    // Simple test data: 5 points
    const MKL_INT n = 5;
    double x[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    double y[] = {0.0, 1.0, 0.0, -1.0, 0.0};

    std::cout << "Input: n=" << n << " points\n";
    std::cout << "x = [";
    for (int i = 0; i < n; ++i)
        std::cout << x[i] << (i < n - 1 ? ", " : "]\n");
    std::cout << "y = [";
    for (int i = 0; i < n; ++i)
        std::cout << y[i] << (i < n - 1 ? ", " : "]\n");

    // Create task
    DFTaskPtr task = nullptr;
    MKL_INT status = dfdNewTask1D(&task, n, x, DF_NON_UNIFORM_PARTITION, 1, y, DF_NO_HINT);

    std::cout << "dfdNewTask1D status: " << status;
    if (status == DF_STATUS_OK)
        std::cout << " (OK)\n";
    else
    {
        std::cout << " (FAILED!)\n";
        return false;
    }

    // Allocate coefficients
    const MKL_INT n_coeffs = 4 * (n - 1);
    std::vector<double> coeffs(n_coeffs);

    // Configure spline
    status = dfdEditPPSpline1D(task, DF_PP_CUBIC, DF_PP_NATURAL,
                               DF_BC_FREE_END, nullptr, DF_NO_IC, nullptr,
                               coeffs.data(), DF_NO_HINT);

    std::cout << "dfdEditPPSpline1D status: " << status;
    if (status == DF_STATUS_OK)
        std::cout << " (OK)\n";
    else
    {
        std::cout << " (FAILED!)\n";
        dfDeleteTask(&task);
        return false;
    }

    // Construct
    status = dfdConstruct1D(task, DF_PP_SPLINE, DF_METHOD_STD);

    std::cout << "dfdConstruct1D status: " << status;
    if (status == DF_STATUS_OK)
        std::cout << " (OK)\n";
    else
    {
        std::cout << " (FAILED!)\n";
        dfDeleteTask(&task);
        return false;
    }

    // Interpolate
    const MKL_INT n_sites = 3;
    double sites[] = {0.5, 1.5, 2.5};
    double results[3];

    status = dfdInterpolate1D(task, DF_INTERP, DF_METHOD_PP,
                              n_sites, sites, DF_NON_UNIFORM_PARTITION,
                              1, nullptr, nullptr,
                              results, DF_NO_HINT, nullptr);

    std::cout << "dfdInterpolate1D status: " << status;
    if (status == DF_STATUS_OK)
        std::cout << " (OK)\n";
    else
    {
        std::cout << " (FAILED!)\n";
        dfDeleteTask(&task);
        return false;
    }

    std::cout << "Interpolated values:\n";
    for (int i = 0; i < n_sites; ++i)
    {
        std::cout << "  f(" << sites[i] << ") = " << results[i] << "\n";
    }

    dfDeleteTask(&task);
    std::cout << "Raw MKL spline: SUCCESS\n\n";
    return true;
}

// Test peak finding
void test_peak_finding()
{
    std::cout << "=== Peak Finding Test ===\n";

    const int n = 100;
    std::vector<double> signal(n);

    // Generate a simple sinusoid
    for (int i = 0; i < n; ++i)
    {
        signal[i] = std::sin(2.0 * M_PI * 5.0 * i / n);
    }

    std::cout << "Signal: sin(2*pi*5*t), n=" << n << "\n";
    std::cout << "Expected ~5 maxima and ~5 minima\n\n";

    // Manual peak finding (same logic as in EEMD)
    std::vector<int> maxima, minima;

    for (int i = 1; i < n - 1; ++i)
    {
        bool left_rise = signal[i] > signal[i - 1];
        bool right_fall = signal[i] > signal[i + 1];
        if (left_rise && right_fall)
        {
            maxima.push_back(i);
        }

        bool left_fall = signal[i] < signal[i - 1];
        bool right_rise = signal[i] < signal[i + 1];
        if (left_fall && right_rise)
        {
            minima.push_back(i);
        }
    }

    std::cout << "Found " << maxima.size() << " maxima: ";
    for (size_t i = 0; i < std::min(size_t(10), maxima.size()); ++i)
    {
        std::cout << maxima[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Found " << minima.size() << " minima: ";
    for (size_t i = 0; i < std::min(size_t(10), minima.size()); ++i)
    {
        std::cout << minima[i] << " ";
    }
    std::cout << "\n\n";
}

// Include EEMD header AFTER our diagnostics
#include "eemd_mkl.hpp"

using namespace eemd;

void test_eemd_components()
{
    std::cout << "=== EEMD Component Test ===\n";

    const int32_t n = 256;
    std::vector<double> signal(n);

    // Generate test signal
    for (int32_t i = 0; i < n; ++i)
    {
        double t = i * 0.01;
        signal[i] = 2.0 * std::sin(2.0 * M_PI * 0.5 * t) + std::sin(2.0 * M_PI * 5.0 * t) + 0.5 * std::sin(2.0 * M_PI * 25.0 * t);
    }

    std::cout << "Signal length: " << n << "\n";

    // Test peak finding with EEMD functions
    std::vector<int32_t> maxima, minima;
    maxima.reserve(n / 2);
    minima.reserve(n / 2);

    find_maxima_noalloc(signal.data(), n, maxima);
    find_minima_noalloc(signal.data(), n, minima);

    std::cout << "EEMD find_maxima: " << maxima.size() << " found\n";
    std::cout << "EEMD find_minima: " << minima.size() << " found\n";

    if (maxima.size() < 2 || minima.size() < 2)
    {
        std::cout << "ERROR: Not enough extrema for EMD!\n\n";
        return;
    }

    // Test boundary extension
    std::vector<double> ext_x, ext_y;
    ext_x.reserve(n + 20);
    ext_y.reserve(n + 20);
    int32_t ext_count = 0, ext_start = 0;

    extend_extrema_noalloc(maxima, signal.data(), n, 2, ext_x, ext_y, ext_count, ext_start);
    std::cout << "Extended extrema: " << ext_count << " points\n";

    // Test MKL spline with EEMD wrapper
    MKLSpline spline;
    bool ok = spline.construct(ext_x.data(), ext_y.data(), ext_count);
    std::cout << "MKLSpline construct: " << (ok ? "SUCCESS" : "FAILED") << "\n";

    if (ok)
    {
        std::vector<double> sites(n), values(n);
        for (int i = 0; i < n; ++i)
            sites[i] = i;

        ok = spline.evaluate(sites.data(), values.data(), n);
        std::cout << "MKLSpline evaluate: " << (ok ? "SUCCESS" : "FAILED") << "\n";
    }

    // Test full EMD
    std::cout << "\n--- Full EMD Test ---\n";
    EEMDConfig config;
    config.max_imfs = 8;

    EEMD decomposer(config);
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;

    ok = decomposer.decompose_emd(signal.data(), n, imfs, residue);
    std::cout << "EMD decompose: " << (ok ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "IMFs extracted: " << imfs.size() << "\n";

    // Test EEMD
    std::cout << "\n--- Full EEMD Test ---\n";
    config.ensemble_size = 10;
    config.noise_std = 0.2;

    EEMD eemd(config);
    int32_t n_imfs = 0;

    ok = eemd.decompose(signal.data(), n, imfs, n_imfs);
    std::cout << "EEMD decompose: " << (ok ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "IMFs extracted: " << n_imfs << "\n\n";
}

int main()
{
    std::cout << "EEMD-MKL Diagnostic Tool\n";
    std::cout << "========================\n\n";

    check_mkl_config();

    if (!test_mkl_spline_raw())
    {
        std::cout << "CRITICAL: Raw MKL spline failed! Check your MKL installation.\n";
        return 1;
    }

    test_peak_finding();
    test_eemd_components();

    std::cout << "=== Diagnostics Complete ===\n";
    return 0;
}