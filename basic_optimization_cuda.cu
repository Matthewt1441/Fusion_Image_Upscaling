#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.cuh"
#include <stdio.h>

const int CHN_NUM = 3;

__global__ void GuassianBlur_Threshold_Map_Kernel(float* blur_map, float* input_map, int width, int height, int radius, float sigma, float threshold)
{
    //Generate Normalized Guassian Kernal for blurring. This may need to be adjusted so I'll make it flexible.
    //We can eventually hardcode this when we settle on ideal blur.
    int kernel_size = 2 * radius + 1;
    int kernel_center = kernel_size / 2;
    float sum = 0.0;
    float guassian_kernel[49] = { 0 };

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    float M_PI = 3.14;

    if (Row < height && Col < width)
    {
        for (int y = 0; y < kernel_size; y++)
        {
            for (int x = 0; x < kernel_size; x++)
            {
                double exponent = -((x - kernel_center) * (x - kernel_center) - (y - kernel_center) * (y - kernel_center)) / (2 * sigma * sigma);
                guassian_kernel[y * kernel_size + x] = exp(exponent) / (2 * M_PI * sigma * sigma);
                sum += guassian_kernel[y * kernel_size + x];
            }
        }
        //Normalize
        //May not want to do this as edge cases will not utilize entire kernel.
        //Will try for now. It may be the right way to do it. I don't know for sure.
        for (int i = 0; i < kernel_size; i++)
            for (int j = 0; j < kernel_size; j++)
                guassian_kernel[i * kernel_size + j] /= sum;

        sum = 0.0;

        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int map_y = Row + i - radius; //
                int map_x = Col + j - radius;

                //If we are within the image
                if (map_x >= 0 && map_x < width && map_y >= 0 && map_y < height) {
                    sum += input_map[map_y * width + map_x] * guassian_kernel[i * kernel_size + j];
                }
            }
        }

        blur_map[Row * width + Col] = (sum > threshold) ? 1.0 : 0.0;
    }
}

__device__ float calculateSSIM_Device(float window1[8][8], float window2[8][8], int window_width, int window_height)
{
    float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, sum12 = 0;
    int size = window_height * window_width;
    int valid_count = 0;

    for (int i = 0; i < window_height; ++i)
    {
        for (int j = 0; j < window_width; ++j)
        {
            if ((window1[i][j] >= 0) && (window2[i][j] >= 0))
            {
                sum1 += window1[i][j];
                sum2 += window2[i][j];
                sum1Sq += window1[i][j] * window1[i][j];
                sum2Sq += window2[i][j] * window2[i][j];
                sum12 += window1[i][j] * window2[i][j];
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
    return ssim;
}

__global__ void Artifact_Grey_Kernel(float* artifact_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    //int window_size = 8;
    //Window size dictates the size of structures that we can detect. Maybe should look into what effect this has
    //on overall image quality & performance
    // Consider the guassian option with an 11x11 window
    float window_img1[8][8] = { 0 };
    float window_img2[8][8] = { 0 };

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    //For now, generate a smaller image.
    if (Row < height && Col < width)
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                if (((Row + i) * width + (Col + j)) < (width * height))
                {
                    window_img1[i][j] = img_1[(Row + i) * width + (Col + j)];
                    window_img2[i][j] = img_2[(Row + i) * width + (Col + j)];
                }
                else
                {
                    window_img1[i][j] = -1;
                    window_img2[i][j] = -1;
                }
            }
        }

        artifact_map[Row * width + Col] = calculateSSIM_Device(window_img1, window_img2, 8, 8) * (float)abs((window_img1[0][0] - window_img2[0][0]) / 255.0);
    }
}

__global__ void nearestNeighbors_GreyCon_Kernel(unsigned char* big_img_data, unsigned char* grey_big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int small_x = 0;    int small_y = 0;

    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;

    if (Row < big_height && Col < big_width)
    {
        small_x = Col / scale;
        small_y = Row / scale;

        r = img_data[3 * (small_y * width + small_x) + 0];
        g = img_data[3 * (small_y * width + small_x) + 1];
        b = img_data[3 * (small_y * width + small_x) + 2];
 
        big_img_data[3 * (Row * big_width + Col) + 0] = r;
        big_img_data[3 * (Row * big_width + Col) + 1] = g;
        big_img_data[3 * (Row * big_width + Col) + 2] = b;

        grey_big_img_data[Row * big_width + Col] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}


__global__ void nearestNeighbors_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int small_x = 0;    int small_y = 0;

    RGBA_t rgba_val;

    if (Row < big_height && Col < big_width)
    {
        small_x = Col / scale;
        small_y = Row / scale;

        rgba_val = img_data[small_y * width + small_x];

        big_img_data[Row * big_width + Col] = rgba_val;

        grey_big_img_data[Row * big_width + Col] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;
    }
}



__device__ float cubicInterpolateDevice_GreyCon(float p[4], float x)
{
    float output = p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));

    output = output * ((output <= 255.0) && (output >= 0.0)) + 255 * (output > 255.0) + 0 * (output < 0);
    return output;
}

__device__ float bicubicInterpolateDevice_GreyCon(float p[4][4], float y, float x)
{
    float arr[4];
    arr[0] = cubicInterpolateDevice_GreyCon(p[0], x);
    arr[1] = cubicInterpolateDevice_GreyCon(p[1], x);
    arr[2] = cubicInterpolateDevice_GreyCon(p[2], x);
    arr[3] = cubicInterpolateDevice_GreyCon(p[3], x);
    return cubicInterpolateDevice_GreyCon(arr, y);
}

__global__ void bicubicInterpolation_GreyCon_Kernel(unsigned char* big_img_data, unsigned char* grey_big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale)
{

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    float window_r[4][4];
    float window_g[4][4];
    float window_b[4][4];

    int sample_x = 0;
    int sample_y = 0;
    
    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;

    if (Row < big_height && Col < big_width)
    {
        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 4; x++)
            {
                window_r[y][x] = 0;
                window_g[y][x] = 0;
                window_b[y][x] = 0;
            }
        }

        if ((Row / scale + 4 < height) && (Col / scale + 4 < width))
        {
            for (int l = 0; l < 4; l++)
            {
                for (int k = 0; k < 4; k++)
                {
                    if ((Row / scale + l < height) && (Col / scale + k < width))
                    {
                        sample_x = Col / scale + k;
                        sample_y = Row / scale + l;

                        if (sample_x > 0)
                            sample_x -= 1;

                        if (sample_y > 0)
                            sample_y -= 1;

                        window_r[l][k] = (float)img_data[3 * (sample_y * width + sample_x) + 0];
                        window_g[l][k] = (float)img_data[3 * (sample_y * width + sample_x) + 1];
                        window_b[l][k] = (float)img_data[3 * (sample_y * width + sample_x) + 2];
                    }

                }
            }

            r = (unsigned char) bicubicInterpolateDevice_GreyCon(window_r, (float)(Row % scale) / scale, (float)(Col % scale) / scale);
            g = (unsigned char) bicubicInterpolateDevice_GreyCon(window_g, (float)(Row % scale) / scale, (float)(Col % scale) / scale);
            b = (unsigned char) bicubicInterpolateDevice_GreyCon(window_b, (float)(Row % scale) / scale, (float)(Col % scale) / scale);

            big_img_data[3 * (Row * big_width + Col) + 0] = r;
            big_img_data[3 * (Row * big_width + Col) + 1] = g;
            big_img_data[3 * (Row * big_width + Col) + 2] = b;
            
            grey_big_img_data[Row * big_width + Col] = 0.21f * r + 0.71f * g + 0.07f * b;
        }
        else
        {
            r = img_data[3 * ((Row / scale) * width + (Col / scale)) + 0];
            g = img_data[3 * ((Row / scale) * width + (Col / scale)) + 1];
            b = img_data[3 * ((Row / scale) * width + (Col / scale)) + 2];

            big_img_data[3 * (Row * big_width + Col) + 0] = r;
            big_img_data[3 * (Row * big_width + Col) + 1] = g;
            big_img_data[3 * (Row * big_width + Col) + 2] = b;

            grey_big_img_data[Row * big_width + Col] = 0.21f * r + 0.71f * g + 0.07f * b;
        }
    }
}


__global__ void bicubicInterpolation_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int input_x = 0;
    int input_y = 0;

    int window_x = 0;
    int window_y = 0;
    
    int output_x = Col;
    int output_y = Row;

    float window_r[4][4];
    float window_g[4][4];
    float window_b[4][4];

    RGBA_t rgba_val;

    for (window_y = 0; window_y < 4; window_y++)
    {
        for (window_x = 0; window_x < 4; window_x++)
        {
            window_r[window_y][window_x] = 0;
            window_g[window_y][window_x] = 0;
            window_b[window_y][window_x] = 0;
        }
    }

    if(output_y < big_height &&  output_x < big_width)
    {
        //Calculate starting index for windows
        float interpolated_x = (float)(output_x / (scale * 1.0));
        float interpolated_y = (float)(output_y / (scale * 1.0));

        int input_block_start_idx_x = (output_x / scale);
        int input_block_start_idx_y = (output_y / scale);

        float dx = interpolated_x - input_block_start_idx_x;
        float dy = interpolated_y - input_block_start_idx_y;

        //We are within a block of the input image, therefore fill windows
        for(window_y = -1; window_y < 3; window_y++)
        {
            for(window_x = -1; window_x < 3; window_x++)
            {
                //Calculate Input Image index
                input_x = input_block_start_idx_x + window_x;
                input_y = input_block_start_idx_y + window_y;

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

                rgba_val = img_data[input_y * width + input_x];

                window_r[window_y + 1][window_x + 1] = (float)rgba_val.r;    //R
                window_g[window_y + 1][window_x + 1] = (float)rgba_val.g;    //G
                window_b[window_y + 1][window_x + 1] = (float)rgba_val.b;    //B
            }
        }

        rgba_val.r = (unsigned char)bicubicInterpolateDevice_GreyCon(window_r, dy, dx);
        rgba_val.g = (unsigned char)bicubicInterpolateDevice_GreyCon(window_g, dy, dx);
        rgba_val.b = (unsigned char)bicubicInterpolateDevice_GreyCon(window_b, dy, dx);

        big_img_data[output_y * big_width + output_x] = rgba_val;

        grey_big_img_data[output_y * big_width + output_x] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;

    }
}
