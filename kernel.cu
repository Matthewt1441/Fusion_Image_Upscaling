#define _USE_MATH_DEFINES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Serial_Kernel.cuh"
//#include "Naive_CUDA_Kernel.cuh"
//#include "Basic_CUDA_Optimize_Kernel.cuh"
#include "Shared_Mem_CUDA_Kernel.cuh"

#include <chrono>

int main()
{
    serialExecution();
    //return naiveCudaExecution();
    //return Code_Testing();
    //return basicCudaOptimizedExecution();
    //sharedMemCudaOptimizedExecution();
    return 1;
}