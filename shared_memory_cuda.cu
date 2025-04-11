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

////Nearest Neighbors but the shared memory uses one thread to output a pixel.
//__global__ void nearestNeighbors_shared_memory_one_thread_per_pixel_Kernel(unsigned char* big_img_data, unsigned char* grey_big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale)
//{
//    extern __shared__ unsigned char img_pixels[];
//
//    int Row = blockIdx.y * blockDim.y + threadIdx.y;
//    int Col = blockIdx.x * blockDim.x + threadIdx.x;
//    
//    int tid_x = threadIdx.x;
//    int tid_y = threadIdx.y;
//
//    int shared_mem_width = blockDim.x / scale;
//    int shared_mem_height = blockDim.y / scale;
//
//    int small_x = 0;    int small_y = 0;
//
//    unsigned char r = 0;
//    unsigned char g = 0;
//    unsigned char b = 0;
//
//    //BLOCK DIM / SCALE THREADS COLLECT DATA FROM GLOBAL MEMORY
//    if (tid_x < shared_mem_width && tid_y < shared_mem_height)
//    {
//        if (Row < big_height && Col < big_width)
//        {
//            small_x = (blockIdx.x * blockDim.x) / scale + tid_x;
//            small_y = (blockIdx.y * blockDim.y) / scale + tid_y;
//
//            img_pixels[CHN_NUM * (tid_y * shared_mem_width + tid_x) + 0] = img_data[CHN_NUM * (small_y * width + small_x) + 0];
//            img_pixels[CHN_NUM * (tid_y * shared_mem_width + tid_x) + 1] = img_data[CHN_NUM * (small_y * width + small_x) + 1];
//            img_pixels[CHN_NUM * (tid_y * shared_mem_width + tid_x) + 2] = img_data[CHN_NUM * (small_y * width + small_x) + 2];
//        }
//
//        else
//        {
//            img_pixels[CHN_NUM * (tid_y * shared_mem_width + tid_x) + 0] = 0;
//            img_pixels[CHN_NUM * (tid_y * shared_mem_width + tid_x) + 1] = 0;
//            img_pixels[CHN_NUM * (tid_y * shared_mem_width + tid_x) + 2] = 0;
//        }
//    }
//
//    __syncthreads();
//
//    //EVERY (VALID) THREAD PARTICPATES IN OUTPUTTING DATA
//    if (Row < big_height && Col < big_width)
//    {
//
//        small_x = tid_x / scale;
//        small_y = tid_y / scale;
//
//        r = img_pixels[CHN_NUM * (small_y * width + small_x) + 0];
//        g = img_pixels[CHN_NUM * (small_y * width + small_x) + 1];
//        b = img_pixels[CHN_NUM * (small_y * width + small_x) + 2];
//
//        big_img_data[CHN_NUM * (Row * big_width + Col) + 0] = r;
//        big_img_data[CHN_NUM * (Row * big_width + Col) + 1] = g;
//        big_img_data[CHN_NUM * (Row * big_width + Col) + 2] = b;
//
//        grey_big_img_data[Row * big_width + Col] = 0.21f * r + 0.71f * g + 0.07f * b;
//    }
//}

//Nearest Neighbors but the shared memory, one thread writes to Scale N Pixels
__global__ void nearestNeighbors_shared_memory_Kernel(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{
    extern __shared__ RGBA_t img_pixels[];

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    int big_x = 0;    int big_y = 0;
    
    RGBA_t rgba_val;

    //BLOCK DIM / SCALE THREADS COLLECT DATA FROM GLOBAL MEMORY
    if (Row < height && Col < width)
    {
        img_pixels[tid_y * blockDim.x + tid_x] = img_data[Row * width + Col];
    }
    //else
    //{
    //    img_pixels[tid_y * blockDim.x + tid_x] = 0;
    //}

    __syncthreads();

    //EVERY (VALID) THREAD PARTICPATES IN OUTPUTTING DATA
    if (Row < height && Col < width)
    {
        for (int y_pix = 0; y_pix < scale; y_pix++)
        {
            for (int x_pix = 0; x_pix < scale; x_pix++)
            {
                big_x = Col * scale + x_pix;
                big_y = Row * scale + y_pix;

                if (Row < big_height && Col < big_width)
                {
                    rgba_val = img_pixels[tid_y * blockDim.x + tid_x];

                    big_img_data[big_y * big_width + big_x] = rgba_val;
                    grey_big_img_data[big_y * big_width + big_x] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;
                }
            }
        }
    }
}


__device__ float cubicInterpolateDevice_Shared(float p[4], float x)
{
    float output = p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));

    output = output * ((output <= 255.0) && (output >= 0.0)) + 255 * (output > 255.0) + 0 * (output < 0);
    return output;
}

__device__ float bicubicInterpolateDevice_Shared(float p[4][4], float x, float y)
{
    float arr[4];
    arr[0] = cubicInterpolateDevice_Shared(p[0], y);
    arr[1] = cubicInterpolateDevice_Shared(p[1], y);
    arr[2] = cubicInterpolateDevice_Shared(p[2], y);
    arr[3] = cubicInterpolateDevice_Shared(p[3], y);
    return cubicInterpolateDevice_Shared(arr, x);
}

//Run with an 8x8 block size
__global__ void bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{
    //Upscaled Image Coordinates (Output)
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float window_r[4][4];
    __shared__ float window_g[4][4];
    __shared__ float window_b[4][4];

    RGBA_t rgba_val;

    //Low Res Image Coordinates (Input)
    //Always read in 4x4 pixels no matter the upscaling factor.
    int input_row = blockIdx.y * 4 + threadIdx.y;
    int input_col = blockIdx.x * 4 + threadIdx.x;

    //Fill shared memory arrays
    if (threadIdx.x < 4 && threadIdx.y < 4)
    {
        rgba_val = img_data[input_row * width + input_col];

        window_r[input_row][input_col] = (float)rgba_val.r;
        window_g[input_row][input_col] = (float)rgba_val.g;
        window_b[input_row][input_col] = (float)rgba_val.b;
    }
    __syncthreads();


    int sample_x = 0;
    int sample_y = 0;

    if (Row < big_height && Col < big_width)
    {
        //What is this checking?
        if ((Row / scale + 4 < height) && (Col / scale + 4 < width))
        {
            //for (int l = 0; l < 4; l++)
            //{
            //    for (int k = 0; k < 4; k++)
            //    {
            //        if ((Row / scale + l < height) && (Col / scale + k < width))
            //        {
            //            sample_x = Col / scale + k;
            //            sample_y = Row / scale + l;

            //            if (sample_x > 0)
            //                sample_x -= 1;

            //            if (sample_y > 0)
            //                sample_y -= 1;

            //            rgba_val = img_data[sample_y * width + sample_x];

            //            window_r[l][k] = (float)rgba_val.r;
            //            window_g[l][k] = (float)rgba_val.g;
            //            window_b[l][k] = (float)rgba_val.b;
            //        }

            //    }
            //}

            rgba_val.r = (unsigned char)bicubicInterpolateDevice_Shared(window_r, (float)(Row % scale) / scale, (float)(Col % scale) / scale);
            rgba_val.g = (unsigned char)bicubicInterpolateDevice_Shared(window_g, (float)(Row % scale) / scale, (float)(Col % scale) / scale);
            rgba_val.b = (unsigned char)bicubicInterpolateDevice_Shared(window_b, (float)(Row % scale) / scale, (float)(Col % scale) / scale);

            big_img_data[Row * big_width + Col] = rgba_val;

            grey_big_img_data[Row * big_width + Col] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;
        }
        else
        {
            rgba_val = img_data[(Row / scale) * width + (Col / scale)];

            big_img_data[Row * big_width + Col] = rgba_val;

            grey_big_img_data[Row * big_width + Col] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;
        }
    }

}


#define WINDOW_SIZE 8

__global__ void Artifact_Shared_Memory_Kernel(float* artifact_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    //int window_size = 8;
    //Window size dictates the size of structures that we can detect. Maybe should look into what effect this has
    //on overall image quality & performance
    // Consider the guassian option with an 11x11 window

    extern __shared__ float window_img[];

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, sum12 = 0;
    float img_diff;

    int valid_count = 0;

    //For now, generate a smaller image.
    
    if (Row < height && Col < width)
    {
        sum1 = img_1[Row * width + Col];    //Using these as temp registers. Just pretend they are called temp1 & temp2
        sum2 = img_2[Row * width + Col];
        window_img[tid_y * WINDOW_SIZE + tid_x + (WINDOW_SIZE * WINDOW_SIZE)] = sum1;
        window_img[tid_y * WINDOW_SIZE + tid_x] = sum2;
        img_diff = (float)abs((sum1 - sum2) / 255.0);
    }
                
    else
    {
        window_img[tid_y * WINDOW_SIZE + tid_x + (WINDOW_SIZE * WINDOW_SIZE)] = -1;
        window_img[tid_y * WINDOW_SIZE + tid_x] = -1;
    }

    sum1 = 0; sum2 = 0; //reset registers

    __syncthreads();

    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            if ((window_img[i * WINDOW_SIZE + j + (WINDOW_SIZE * WINDOW_SIZE)] >= 0) && (window_img[i * WINDOW_SIZE + j] >= 0))
            {
                sum1 += window_img[i * WINDOW_SIZE + j + (WINDOW_SIZE * WINDOW_SIZE)];
                sum2 += window_img[i * WINDOW_SIZE + j];
                sum1Sq += window_img[i * WINDOW_SIZE + j + (WINDOW_SIZE * WINDOW_SIZE)] * window_img[i * WINDOW_SIZE + j + (WINDOW_SIZE * WINDOW_SIZE)];
                sum2Sq += window_img[i * WINDOW_SIZE + j] * window_img[i * WINDOW_SIZE + j];
                sum12 += window_img[i * WINDOW_SIZE + j + (WINDOW_SIZE * WINDOW_SIZE)] * window_img[i * WINDOW_SIZE + j];
                valid_count++;
            }
        }
    }

    float mu1 = sum1 / valid_count;
    float mu2 = sum2 / valid_count;
    float sigma1Sq = (sum1Sq / valid_count) - (mu1 * mu1);
    float sigma2Sq = (sum2Sq / valid_count) - (mu2 * mu2);
    float sigma12 = (sum12 / valid_count) - (mu1 * mu2);

    // Stabilizing constants
    float C1 = 6.5025; // (K1*L)^2, where K1=0.01 and L=255
    float C2 = 58.5225; // (K2*L)^2, where K2=0.03 and L=255

    float ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1Sq + sigma2Sq + C2));

    artifact_map[Row * width + Col] = ssim * img_diff;
}