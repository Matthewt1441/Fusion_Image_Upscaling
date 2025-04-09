#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.cuh"

const int CHN_NUM = 3;

//Initial Naive approach
__global__ void rgbToRGBA_Kernel(RGBA_t* d_RGBA_img, unsigned char* d_rgb_img, int numpixels)
{
    // Each thread processes one pixel
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int sharedIdx = tid * 3;

    // Shared memory for the current block of RGB values
    __shared__ unsigned char sharedRGB[256 * 3];  // Assuming block size is 256 threads

    // Load data into shared memory
    if (idx < numpixels) 
    {
        sharedRGB[sharedIdx + 0] = d_rgb_img[idx * 3 + 0];
        sharedRGB[sharedIdx + 1] = d_rgb_img[idx * 3 + 1];
        sharedRGB[sharedIdx + 2] = d_rgb_img[idx * 3 + 2];
    }

    // Synchronize to ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Now process the RGB to RGBA conversion in shared memory
    if (idx < numpixels) {
        // Read RGB values from shared memory
        unsigned char r = sharedRGB[sharedIdx + 0];
        unsigned char g = sharedRGB[sharedIdx + 1];
        unsigned char b = sharedRGB[sharedIdx + 2];

        // Write to RGBA array (global memory)
        d_RGBA_img[idx].r = r;
        d_RGBA_img[idx].g = g;
        d_RGBA_img[idx].b = b;
        d_RGBA_img[idx].a = 255;  // Alpha is fully opaque
    }
}


//Initial Naive approach
__global__ void rgbaToRGB_Kernel(unsigned char* d_rgb_img, RGBA_t* d_rgba_img, int numpixels)
{
    // Each thread processes one pixel
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int sharedIdx = threadIdx.x;

    // Shared memory for the current block of RGBA values
    __shared__ RGBA_t sharedRGB[256];  // Assuming block size is 256 threads

    // Load data into shared memory
    if (idx < numpixels)
    {
        sharedRGB[sharedIdx] = d_rgba_img[idx];
    }

    // Synchronize to ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Now process the RGBA to RGB conversion in shared memory
    if (idx < numpixels) {
        // Read RGBA values from shared memory
        RGBA_t rgba_val = sharedRGB[sharedIdx];

        // Write to RGB array (global memory)
        d_rgb_img[idx * 3 + 0] = rgba_val.r;
        d_rgb_img[idx * 3 + 1] = rgba_val.g;
        d_rgb_img[idx * 3 + 2] = rgba_val.b;
    }
}

//Nearest Neighbors but the shared memory uses one thread to output a pixel.
__global__ void nearestNeighbors_shared_memory_one_thread_per_pixel_Kernel(unsigned char* big_img_data, unsigned char* grey_big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale)
{
    extern __shared__ unsigned char img_pixels[];

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    int shared_mem_width = blockDim.x / scale;
    int shared_mem_height = blockDim.y / scale;

    int small_x = 0;    int small_y = 0;

    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;

    //BLOCK DIM / SCALE THREADS COLLECT DATA FROM GLOBAL MEMORY
    if (tid_x < shared_mem_width && tid_y < shared_mem_height)
    {
        if (Row < big_height && Col < big_width)
        {
            small_x = (blockIdx.x * blockDim.x) / scale + tid_x;
            small_y = (blockIdx.y * blockDim.y) / scale + tid_y;

            img_pixels[CHN_NUM * (tid_y * shared_mem_width + tid_x) + 0] = img_data[CHN_NUM * (small_y * width + small_x) + 0];
            img_pixels[CHN_NUM * (tid_y * shared_mem_width + tid_x) + 1] = img_data[CHN_NUM * (small_y * width + small_x) + 1];
            img_pixels[CHN_NUM * (tid_y * shared_mem_width + tid_x) + 2] = img_data[CHN_NUM * (small_y * width + small_x) + 2];
        }

        else
        {
            img_pixels[CHN_NUM * (tid_y * shared_mem_width + tid_x) + 0] = 0;
            img_pixels[CHN_NUM * (tid_y * shared_mem_width + tid_x) + 1] = 0;
            img_pixels[CHN_NUM * (tid_y * shared_mem_width + tid_x) + 2] = 0;
        }
    }

    __syncthreads();

    //EVERY (VALID) THREAD PARTICPATES IN OUTPUTTING DATA
    if (Row < big_height && Col < big_width)
    {

        small_x = tid_x / scale;
        small_y = tid_y / scale;

        r = img_pixels[CHN_NUM * (small_y * width + small_x) + 0];
        g = img_pixels[CHN_NUM * (small_y * width + small_x) + 1];
        b = img_pixels[CHN_NUM * (small_y * width + small_x) + 2];

        big_img_data[CHN_NUM * (Row * big_width + Col) + 0] = r;
        big_img_data[CHN_NUM * (Row * big_width + Col) + 1] = g;
        big_img_data[CHN_NUM * (Row * big_width + Col) + 2] = b;

        grey_big_img_data[Row * big_width + Col] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

