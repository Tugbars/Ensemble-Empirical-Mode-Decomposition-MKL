/**
 * Test Akima spline construction with MKL
 * Run this to debug DF_PP_AKIMA parameters
 */

#include <mkl.h>
#include <mkl_df.h>
#include <cstdio>
#include <vector>

int main() {
    printf("=== Akima Spline MKL Test ===\n\n");
    
    // Simple test data: 10 points
    const int n = 10;
    double x[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    double y[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    
    std::vector<double> coeffs(4 * (n - 1));
    
    // Test 1: Cubic spline (should work)
    {
        DFTaskPtr task = nullptr;
        MKL_INT status = dfdNewTask1D(&task, n, x, DF_NON_UNIFORM_PARTITION, 1, y, DF_NO_HINT);
        printf("Cubic - NewTask: %lld\n", (long long)status);
        
        status = dfdEditPPSpline1D(task, DF_PP_CUBIC, DF_PP_NATURAL,
                                   DF_BC_FREE_END, nullptr, DF_NO_IC, nullptr,
                                   coeffs.data(), DF_NO_HINT);
        printf("Cubic - EditPPSpline: %lld\n", (long long)status);
        
        status = dfdConstruct1D(task, DF_PP_SPLINE, DF_METHOD_STD);
        printf("Cubic - Construct: %lld\n", (long long)status);
        
        dfDeleteTask(&task);
        printf("Cubic: %s\n\n", status == DF_STATUS_OK ? "SUCCESS" : "FAILED");
    }
    
    // Test 2: Akima spline - attempt 1
    {
        DFTaskPtr task = nullptr;
        MKL_INT status = dfdNewTask1D(&task, n, x, DF_NON_UNIFORM_PARTITION, 1, y, DF_NO_HINT);
        printf("Akima1 - NewTask: %lld\n", (long long)status);
        
        // Try: s_order=DF_PP_AKIMA, s_type=DF_PP_DEFAULT, bc=DF_NO_BC
        status = dfdEditPPSpline1D(task, DF_PP_AKIMA, DF_PP_DEFAULT,
                                   DF_NO_BC, nullptr, DF_NO_IC, nullptr,
                                   coeffs.data(), DF_NO_HINT);
        printf("Akima1 - EditPPSpline: %lld\n", (long long)status);
        
        if (status == DF_STATUS_OK) {
            status = dfdConstruct1D(task, DF_PP_SPLINE, DF_METHOD_STD);
            printf("Akima1 - Construct: %lld\n", (long long)status);
        }
        
        dfDeleteTask(&task);
        printf("Akima1 (DEFAULT/NO_BC): %s\n\n", status == DF_STATUS_OK ? "SUCCESS" : "FAILED");
    }
    
    // Test 3: Akima spline - attempt 2
    {
        DFTaskPtr task = nullptr;
        MKL_INT status = dfdNewTask1D(&task, n, x, DF_NON_UNIFORM_PARTITION, 1, y, DF_NO_HINT);
        printf("Akima2 - NewTask: %lld\n", (long long)status);
        
        // Try: s_order=DF_PP_AKIMA, s_type=DF_PP_AKIMA, bc=DF_NO_BC
        status = dfdEditPPSpline1D(task, DF_PP_AKIMA, DF_PP_AKIMA,
                                   DF_NO_BC, nullptr, DF_NO_IC, nullptr,
                                   coeffs.data(), DF_NO_HINT);
        printf("Akima2 - EditPPSpline: %lld\n", (long long)status);
        
        if (status == DF_STATUS_OK) {
            status = dfdConstruct1D(task, DF_PP_SPLINE, DF_METHOD_STD);
            printf("Akima2 - Construct: %lld\n", (long long)status);
        }
        
        dfDeleteTask(&task);
        printf("Akima2 (AKIMA/NO_BC): %s\n\n", status == DF_STATUS_OK ? "SUCCESS" : "FAILED");
    }
    
    // Test 4: Check if DF_PP_AKIMA is defined
    printf("DF_PP_AKIMA value: %d\n", DF_PP_AKIMA);
    printf("DF_PP_CUBIC value: %d\n", DF_PP_CUBIC);
    printf("DF_PP_LINEAR value: %d\n", DF_PP_LINEAR);
    
    return 0;
}
