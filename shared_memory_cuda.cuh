#pragma once
#include "util.cuh"

extern __constant__ float d_bic_kernel[64];

__device__ float cubicInterpolateDevice_Shared(float p[4], float x);
__device__ float bicubicInterpolateDevice_Shared(float p[4][4], float x, float y);
__global__ void bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale);
//__global__ void nearestNeighbors_shared_memory_one_thread_per_pixel_Kernel(unsigned char* big_img_data, unsigned char* grey_big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale);
__global__ void nearestNeighbors_shared_memory_Kernel(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale);
__global__ void Artifact_Shared_Memory_Kernel(float* artifact_map, unsigned char* img_1, unsigned char* img_2, int width, int height);

__global__ void nearestNeighbors_shared_memory_one_thread_per_pixel_Kernel(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale);

__global__ void horizontalBicubicConvolve( RGBA_t* big_img_data, RGBA_t* img_data, float* kernel, int big_width, int big_height, int width, int height, int scale, int ksize);
__global__ void verticalBicubicConvolve( RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, float* kernel,int big_width, int big_height, int width, int height, int scale, int ksize);

__global__ void GuassianBlur_Threshold_Map_Shared_Memory_Kernel(float* blur_map, float* input_map, float* kernel, int width, int height, float threshold, int ksize);