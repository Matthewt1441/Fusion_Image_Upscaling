#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

__device__ float cubicInterpolateDevice_GreyCon(float p[4], float x)
{
    float output = p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));

    output = output * ((output <= 255.0) && (output >= 0.0)) + 255 * (output > 255.0) + 0 * (output < 0);
    return output;
}

__device__ float bicubicInterpolateDevice_GreyCon(float p[4][4], float x, float y)
{
    float arr[4];
    arr[0] = cubicInterpolateDevice_GreyCon(p[0], y);
    arr[1] = cubicInterpolateDevice_GreyCon(p[1], y);
    arr[2] = cubicInterpolateDevice_GreyCon(p[2], y);
    arr[3] = cubicInterpolateDevice_GreyCon(p[3], y);
    return cubicInterpolateDevice_GreyCon(arr, x);
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
