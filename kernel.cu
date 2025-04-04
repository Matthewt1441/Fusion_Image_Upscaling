#define _USE_MATH_DEFINES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Serial_Kernel.cuh"
#include "Naive_CUDA_Kernel.cuh"
#include "Basic_CUDA_Optimize_Kernel.cuh"

#include <chrono>

#include <SDL.h>
#undef main
#include <SDL_ttf.h>
#undef main

int main()
{
    //return serialExecution();
    //return naiveCudaExecution();
    //return Code_Testing();
    return basicCudaOptimizedExecution();

}