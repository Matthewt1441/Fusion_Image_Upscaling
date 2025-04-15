#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.cuh"
#include <stdio.h>

const int CHN_NUM = 3;

//__constant__ float d_bic_kernel[64];

////Nearest Neighbors but the shared memory uses one thread to output a pixel.
__global__ void nearestNeighbors_shared_memory_one_thread_per_pixel_Kernel(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{
    extern __shared__ RGBA_t img_pixels[];

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int SHARE_MEM_WIDTH = blockDim.x / scale;

    int big_x = 0;    int big_y = 0;

    RGBA_t rgba_val;

    //BLOCK DIM / SCALE THREADS COLLECT DATA FROM GLOBAL MEMORY
    if (Row < big_height && Col < big_width)
    {
        if (tid_y < SHARE_MEM_WIDTH && tid_x < SHARE_MEM_WIDTH)
        {
            img_pixels[tid_y * (SHARE_MEM_WIDTH) + tid_x] = img_data[((blockIdx.y * blockDim.y)/scale + tid_y) * width + ((blockIdx.x * blockDim.x)/scale) + tid_x];
        }
    }
    //else
    //{
    //    img_pixels[tid_y * blockDim.x + tid_x] = 0;
    //}

    __syncthreads();

    //EVERY (VALID) THREAD PARTICPATES IN OUTPUTTING DATA
    if (Row < big_height && Col < big_width)
    {
        rgba_val = img_pixels[(tid_y / scale) * SHARE_MEM_WIDTH + (tid_x / scale)];

        big_img_data[Row * big_width + Col] = rgba_val;
        grey_big_img_data[Row * big_width + Col] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;
   
    }
}

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

__device__ float bicubicInterpolateDevice_Shared(float p[4][4], float y, float x)
{
    float arr[4];
    arr[0] = cubicInterpolateDevice_Shared(p[0], x);
    arr[1] = cubicInterpolateDevice_Shared(p[1], x);
    arr[2] = cubicInterpolateDevice_Shared(p[2], x);
    arr[3] = cubicInterpolateDevice_Shared(p[3], x);
    return cubicInterpolateDevice_Shared(arr, y);
}

//Run with an 8x8 block size
__global__ void bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int input_x = 0;
    int input_y = 0;
    
    int window_x = 0;
    int window_y = 0;

    int output_x = Col;
    int output_y = Row;

    //Assume scale of 2 for now
    __shared__ RGBA_t s_intput_tile[11][11];

    float window_r[4][4];
    float window_g[4][4];
    float window_b[4][4];

    RGBA_t rgba_val;

    if(threadIdx.x < 11 && threadIdx.y < 11)
    {
        input_x = (blockIdx.x * 8 + threadIdx.x) - 1;
        input_y = (blockIdx.y * 8 + threadIdx.y) - 1;

        // Fill window with Nearest Neighbor edge behavior
        if(input_x < 0 || input_x >= width)
        {
            // Find nearest in-bounds pixel
            input_x = (input_x < 0) ? 0 : width - 1;
        }
        // Fill window with Nearest Neighbor edge behavior
        if(input_y < 0 || input_y >= height)
        {
            // Find nearest in-bounds pixel
            input_y = (input_y < 0) ? 0 : height - 1;
        }

        s_intput_tile[threadIdx.y][threadIdx.x] = img_data[input_y * width + input_x];
    }
    __syncthreads();

    if(output_y < big_height && output_x < big_width)
    {
        //Calculate starting index for windows
        float interpolated_x = (float)((threadIdx.x / (scale * 1.0)) + 1.0);
        float interpolated_y = (float)((threadIdx.y / (scale * 1.0)) + 1.0);

        int input_block_start_idx_x = (threadIdx.x / scale) + 1;
        int input_block_start_idx_y = (threadIdx.y / scale) + 1;

        float dx = interpolated_x - input_block_start_idx_x;
        float dy = interpolated_y - input_block_start_idx_y;

        for(window_y = -1; window_y < 3; window_y++)
        {
            for(window_x = -1; window_x < 3; window_x++)
            {
                //Calculate Input Image index
                input_x = input_block_start_idx_x + window_x;
                input_y = input_block_start_idx_y + window_y;

                rgba_val = s_intput_tile[input_y][input_x];

                window_r[window_y + 1][window_x + 1] = (float)rgba_val.r;    //R
                window_g[window_y + 1][window_x + 1] = (float)rgba_val.g;    //G
                window_b[window_y + 1][window_x + 1] = (float)rgba_val.b;    //B
            }
        }

        rgba_val.r = (unsigned char)bicubicInterpolateDevice_Shared(window_r, dy, dx);
        rgba_val.g = (unsigned char)bicubicInterpolateDevice_Shared(window_g, dy, dx);
        rgba_val.b = (unsigned char)bicubicInterpolateDevice_Shared(window_b, dy, dx);

        big_img_data[output_y * big_width + output_x] = rgba_val;

        grey_big_img_data[output_y * big_width + output_x] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;

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

__global__ void horizontalBicubicConvolve( RGBA_t* big_img_data, RGBA_t* img_data, float* kernel, int big_width, int big_height, int width, int height, int scale, int ksize) 
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int output_x = Col;
    int output_y = Row;

    int input_x = 0;
    int input_y = 0;

    float sum_r = 0.0;
    float sum_g = 0.0;
    float sum_b = 0.0;
    int kRadius = ksize / 2;

    RGBA_t rgba_val;

    if(Row == 0 && Col == 0)
    {
        printf("BIC Kernel\n");
        for(int i = 0; i < ksize; i++)
        {
            
            printf("%f, ",  kernel[i]);
        }
        printf("\n");
    }

    if(output_y < big_height &&  output_x < big_width)
    {
        int input_block_start_idx_x = (output_x / scale);
        int input_block_start_idx_y = (output_y / scale);
        input_y = input_block_start_idx_y;

        for(int k = -kRadius; k <= kRadius; ++k)
        {
            //Calculate Input Image index
            input_x = input_block_start_idx_x + k;

            // Fill window with Nearest Neighbor edge behavior
            if(input_x < 0 || input_x >= width)
            {
                // Find nearest in-bounds pixel
                input_x = (input_x < 0) ? 0 : width - 1;
            }

            rgba_val = img_data[input_y * width + input_x];

            sum_r += kernel[k + kRadius] * (1.0)*rgba_val.r;
            sum_g += kernel[k + kRadius] * (1.0)*rgba_val.g;
            sum_b += kernel[k + kRadius] * (1.0)*rgba_val.b;
        }

        rgba_val.r = sum_r;
        rgba_val.g = sum_g;
        rgba_val.b = sum_b;
        
        big_img_data[output_y * big_width + output_x] = rgba_val;
    }
}

__global__ void verticalBicubicConvolve( RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, float* kernel, int big_width, int big_height, int width, int height, int scale, int ksize) 
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int output_x = Col;
    int output_y = Row;

    int input_x = 0;
    int input_y = 0;

    float sum_r = 0.0;
    float sum_g = 0.0;
    float sum_b = 0.0;
    int kRadius = ksize / 2;

    RGBA_t rgba_val;

    if(output_y < big_height &&  output_x < big_width)
    {
        int input_block_start_idx_x = (output_x / scale);
        int input_block_start_idx_y = (output_y / scale);
        input_x = input_block_start_idx_x;

        for(int k = -kRadius; k <= kRadius; ++k)
        {
            //Calculate Input Image index
            input_y = input_block_start_idx_y + k;

            // Fill window with Nearest Neighbor edge behavior
            if(input_y < 0 || input_y >= height)
            {
                // Find nearest in-bounds pixel
                input_y = (input_y < 0) ? 0 : height - 1;
            }

            rgba_val = img_data[input_y * width + input_x];

            sum_r += kernel[k + kRadius] * (1.0)*rgba_val.r;
            sum_g += kernel[k + kRadius] * (1.0)*rgba_val.g;
            sum_b += kernel[k + kRadius] * (1.0)*rgba_val.b;
        }

        rgba_val.r = sum_r;
        rgba_val.g = sum_g;
        rgba_val.b = sum_b;
        
        big_img_data[output_y * big_width + output_x] = rgba_val;
        
        grey_big_img_data[output_y * big_width + output_x] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;

    }
}