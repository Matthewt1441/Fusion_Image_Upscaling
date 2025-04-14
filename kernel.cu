#define _USE_MATH_DEFINES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "Serial_Kernel.cuh"
//#include "Naive_CUDA_Kernel.cuh"
//#include "Basic_CUDA_Optimize_Kernel.cuh"
//#include "Shared_Mem_CUDA_Kernel.cuh"
#include "NN_Test.cuh"

#include <chrono>

int main()
{
    //serialExecution();
    //return naiveCudaExecution();
    //return Code_Testing();
    //return basicCudaOptimizedExecution();
    //return sharedMemCudaOptimizedExecution();
    NN_Execution_Test("./LM_Frame/image%d.ppm", 2, 16);

    return 0;
}