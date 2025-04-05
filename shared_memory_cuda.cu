#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int CHN_NUM = 3;

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

