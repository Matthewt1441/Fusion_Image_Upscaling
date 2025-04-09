#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.cuh"

__global__ void GuassianBlur_Threshold_Map_Kernel(float* blur_map, float* input_map, int width, int height, int radius, float sigma, float threshold);
__global__ void Artifact_Grey_Kernel(float* artifact_map, unsigned char* img_1, unsigned char* img_2, int width, int height);
__global__ void bicubicInterpolation_GreyCon_Kernel(unsigned char* big_img_data, unsigned char* grey_big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale);
__global__ void nearestNeighbors_GreyCon_Kernel(unsigned char* big_img_data, unsigned char* grey_big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale);


__global__ void bicubicInterpolation_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale);
__global__ void nearestNeighbors_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale);
