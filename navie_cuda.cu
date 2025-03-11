#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "naive_cuda.cuh"

__global__ void RGB2GreyscaleKernel(unsigned char* rgb_img, unsigned char* grey_img, int width, int height)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int rgbidx = rgbidx = 3 * (Row * width + Col);
    grey_img[Row * width + Col] = 0.21f * rgb_img[rgbidx + 0] + 0.71f * rgb_img[rgbidx + 1] + 0.07f * rgb_img[rgbidx + 2];
    
}

__global__ void nearestNeighborsKernel(unsigned char* big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;                    
    int Col = blockIdx.x * blockDim.x + threadIdx.x;                      

    int small_x = 0;    int small_y = 0;

    if (Row < big_height && Col < big_width)
    {
        small_x = Col / scale;
        small_y = Row / scale;

        big_img_data[3 * (Row * big_width + Col) + 0] = img_data[3 * (small_y * width + small_x) + 0];
        big_img_data[3 * (Row * big_width + Col) + 1] = img_data[3 * (small_y * width + small_x) + 1];
        big_img_data[3 * (Row * big_width + Col) + 2] = img_data[3 * (small_y * width + small_x) + 2];
    }
}

__device__ float cubicInterpolateDevice(float p[4], float x)
{
    float output = p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));

    output = output * ((output <= 255.0) && (output >= 0.0)) + 255 * (output > 255.0) + 0 * (output < 0);
    return output;
}

__device__ float bicubicInterpolateDevice(float p[4][4], float x, float y)
{
    float arr[4];
    arr[0] = cubicInterpolateDevice(p[0], y);
    arr[1] = cubicInterpolateDevice(p[1], y);
    arr[2] = cubicInterpolateDevice(p[2], y);
    arr[3] = cubicInterpolateDevice(p[3], y);
    return cubicInterpolateDevice(arr, x);
}

__global__ void bicubicInterpolationKernel(unsigned char* big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale)
{

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    float window_r[4][4];
    float window_g[4][4];
    float window_b[4][4];

    int sample_x = 0;
    int sample_y = 0;

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
                    //window_r[l][k] = (float)img_data[3 * ((l + Row / scale) * width + Col / scale + k) + 0];
                    //window_g[l][k] = (float)img_data[3 * ((l + Row / scale) * width + Col / scale + k) + 1];
                    //window_b[l][k] = (float)img_data[3 * ((l + Row / scale) * width + Col / scale + k) + 2];

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

        float temp1 = bicubicInterpolateDevice(window_r, (float)(Row % scale) / scale, (float)(Col % scale) / scale);
        float temp2 = bicubicInterpolateDevice(window_g, (float)(Row % scale) / scale, (float)(Col % scale) / scale);
        float temp3 = bicubicInterpolateDevice(window_b, (float)(Row % scale) / scale, (float)(Col % scale) / scale);

        big_img_data[3 * (Row * big_width + Col) + 0] = (unsigned char)temp1;
        big_img_data[3 * (Row * big_width + Col) + 1] = (unsigned char)temp2;
        big_img_data[3 * (Row * big_width + Col) + 2] = (unsigned char)temp3;
    }
    else
    {
        big_img_data[3 * (Row * big_width + Col) + 0] = img_data[3 * ((Row / scale) * width + (Col / scale)) + 0];
        big_img_data[3 * (Row * big_width + Col) + 1] = img_data[3 * ((Row / scale) * width + (Col / scale)) + 1];
        big_img_data[3 * (Row * big_width + Col) + 2] = img_data[3 * ((Row / scale) * width + (Col / scale)) + 2];
    }
}
