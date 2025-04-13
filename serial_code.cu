#define _USE_MATH_DEFINES

#include "serial_code.cuh"
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <stdio.h>

const int CHN_NUM = 3;

float calculateSSIM(float window1[8][8], float window2[8][8], int window_width, int window_height) {
    float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, sum12 = 0;
    int size = window_height * window_width;
    int valid_count = 0;

    for (int i = 0; i < window_height; ++i) {
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

void SSIM_Grey(float* ssim_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    //int window_size = 8;
    //Window size dictates the size of structures that we can detect. Maybe should look into what effect this has
    //on overall image quality & performance
    // Consider the guassian option with an 11x11 window
    float window_img1[8][8] = { 0 };
    float window_img2[8][8] = { 0 };

    int x_blocks = (width - 1) / 8 + 1;
    int y_blocks = (height - 1) / 8 + 1;

    float ssim_num = 0;

    //For now, generate a smaller image.
    for (int y_blk = 0; y_blk < y_blocks; y_blk++)
    {
        for (int x_blk = 0; x_blk < x_blocks; x_blk++)
        {
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    if ((x_blk * 8 + j) < (width) && ((y_blk * 8 + i) * height))
                    {
                        window_img1[i][j] = img_1[(y_blk * 8 + i) * width + (x_blk * 8 + j)];
                        window_img2[i][j] = img_2[(y_blk * 8 + i) * width + (x_blk * 8 + j)];
                    }
                    else
                    {
                        window_img1[i][j] = -1;
                        window_img2[i][j] = -1;
                    }
                }
            }

            ssim_num = calculateSSIM(window_img1, window_img2, 8, 8);

            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    if ((x_blk * 8 + j) < (width) && ((y_blk * 8 + i) * height))
                    {
                        ssim_map[(y_blk * 8 + i) * width + (x_blk * 8 + j)] = ssim_num;
                    }
                }
            }
        }
    }
}

void ABS_Difference_Grey(float* diff_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{

    float img_1_signed = 0;
    float img_2_signed = 0;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            img_1_signed = (float)img_1[y * width + x];
            img_2_signed = (float)img_2[y * width + x];
            diff_map[y * width + x] = (float)abs((img_1_signed - img_2_signed) / 255.0); //Normalize 
        }
    }
}

void RGB2Greyscale(unsigned char* grey_img, unsigned char* rgb_img, int width, int height)
{
    int rgbidx = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            rgbidx = 3 * (y * width + x);
            grey_img[y * width + x] = 0.21f * rgb_img[rgbidx + 0] + 0.71f * rgb_img[rgbidx + 1] + 0.07f * rgb_img[rgbidx + 2];
        }
    }
}

void Map2Greyscale(unsigned char* grey_img, float* map, int width, int height, int scale)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            grey_img[y * width + x] = (unsigned char)(scale * map[y * width + x]);
        }
    }
}

void MapMul(float* product_map, float* map_1, float* map_2, int width, int height)
{
    int idx = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            idx = y * width + x;
            product_map[idx] = map_1[idx] * map_2[idx];
        }
    }
}


void MapThreshold(float* map, float threshold, int width, int height)
{
    int idx = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            idx = y * width + x;
            if (map[idx] > threshold)
            {
                map[idx] = 1.0;
            }
            else
            {
                map[idx] = 0.0;
            }
        }
    }
}

void GuassianBlur_Map(float* blur_map, float* input_map, int width, int height, int radius, float sigma)
{
    //Generate Normalized Guassian Kernal for blurring. This may need to be adjusted so I'll make it flexible.
    //We can eventually hardcode this when we settle on ideal blur.
    int kernel_size = 2 * radius + 1;
    int kernel_center = kernel_size / 2;
    float sum = 0.0;
    float* guassian_kernel = (float*)malloc(sizeof(float) * kernel_size * kernel_size);

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
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            guassian_kernel[i * kernel_size + j] /= sum;
        }
    }

    //Run through image with kernel centered at current pixel
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            sum = 0.0;
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    int map_y = y + i - radius; //
                    int map_x = x + j - radius;

                    //If we are within the image
                    if (map_x >= 0 && map_x < width && map_y >= 0 && map_y < height) {
                        sum += input_map[map_y * width + map_x] * guassian_kernel[i * kernel_size + j];
                    }
                }
            }
            blur_map[y * width + x] = sum;
        }
    }

    free(guassian_kernel);

}

void GuassianBlur_Img(unsigned char* blur_img, unsigned char* input_img, int width, int height, int radius, float sigma)
{
    //Generate Normalized Guassian Kernal for blurring. This may need to be adjusted so I'll make it flexible.
    //We can eventually hardcode this when we settle on ideal blur.
    int kernel_size = 2 * radius + 1;
    int kernel_center = kernel_size / 2;
    float sum = 0.0;
    float* guassian_kernel = (float*)malloc(sizeof(float) * kernel_size * kernel_size);

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
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            guassian_kernel[i * kernel_size + j] /= sum;
        }
    }

    //Run through image with kernel centered at current pixel
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            sum = 0.0;
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    int map_y = y + i - radius; //
                    int map_x = x + j - radius;

                    //If we are within the image
                    if (map_x >= 0 && map_x < width && map_y >= 0 && map_y < height) {
                        sum += (float)input_img[map_y * width + map_x] * guassian_kernel[i * kernel_size + j];
                    }
                }
            }
            blur_img[y * width + x] = (unsigned char)sum;
        }
    }

    free(guassian_kernel);
}

void Image_Fusion(unsigned char* fused_img, unsigned char* img_1, unsigned char* img_2, float* weight_map, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int map_idx = (y * width + x);
            int img_idx = 3 * map_idx;
            fused_img[img_idx + 0] = img_1[img_idx + 0] * weight_map[map_idx] + img_2[img_idx + 0] * (1.0 - weight_map[map_idx]);
            fused_img[img_idx + 1] = img_1[img_idx + 1] * weight_map[map_idx] + img_2[img_idx + 1] * (1.0 - weight_map[map_idx]);
            fused_img[img_idx + 2] = img_1[img_idx + 2] * weight_map[map_idx] + img_2[img_idx + 2] * (1.0 - weight_map[map_idx]);
        }
    }
}

void nearestNeighbors(unsigned char* big_img_data, int big_width, int big_height, unsigned char* img_data, int width, int height, int scale)
{
    int small_x, small_y;

    for (int y = 0; y < big_height; y++)
    {
        for (int x = 0; x < big_width; x++)
        {
            small_x = x / scale;
            small_y = y / scale;

            big_img_data[3 * (y * big_width + x) + 0] = img_data[3 * (small_y * width + small_x) + 0];
            big_img_data[3 * (y * big_width + x) + 1] = img_data[3 * (small_y * width + small_x) + 1];
            big_img_data[3 * (y * big_width + x) + 2] = img_data[3 * (small_y * width + small_x) + 2];
        }
    }
}

float cubicInterpolate(float p[4], float x) 
{
    float output = p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));

    if ((output <= 255.0) && (output >= 0.0))
    {
        return output;
    }
    else if (output > 255.0)
    {
        return 255;
    }
    return 0.0;
}

//                               y  x
float bicubicInterpolate(float p[4][4], float x, float y) 
{
    float arr[4];
    arr[0] = cubicInterpolate(p[0], y);
    arr[1] = cubicInterpolate(p[1], y);
    arr[2] = cubicInterpolate(p[2], y);
    arr[3] = cubicInterpolate(p[3], y);
    return cubicInterpolate(arr, x);
}

void bicubicInterpolation(unsigned char* big_img_data, int big_width, int big_height, unsigned char* img_data, int width, int height, int scale)
{
    float window_r[4][4];
    float window_g[4][4];
    float window_b[4][4];

    int f = scale;
    int w = width;
    int h = height;

    int sample_x = 0;//
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

    //For y within Big image size
    for (int y = 0; y < f * h; y++)
    {
        //For x within big image size
        for (int x = 0; x < f * w; x++)
        {

            //Check if y & x divided by the scale is within small image size & not at the edge
            if ((y / f + 4  < h) && (x / f + 4 < w))
            {
                //4x4 window loop
                for (int l = 0; l < 4; l++) //Y
                {
                    //4x4 window loop
                    for (int k = 0; k < 4; k++) //X
                    {
                        ////This check is not needed as its already done above
                        //if ((y / f + l < h) && (x / f + k < w))
                        //{
                            sample_x = x / f + k;
                            sample_y = y / f + l;

                            //if (sample_x > 0)
                            //    sample_x-=1;

                            //if (sample_y > 0)
                            //    sample_y-=1;

                            window_r[l][k] = (float)img_data[3 * (sample_y * width + sample_x) + 0];
                            window_g[l][k] = (float)img_data[3 * (sample_y * width + sample_x) + 1];
                            window_b[l][k] = (float)img_data[3 * (sample_y * width + sample_x) + 2];
                        //}
                    }
                }

                //float temp1 = bicubicInterpolate(window_r, (float)(y % (4*f))/(4*f), (float)(x % (4*f))/(4*f));
                //float temp2 = bicubicInterpolate(window_g, (float)(y % (4*f))/(4*f), (float)(x % (4*f))/(4*f));
                //float temp3 = bicubicInterpolate(window_b, (float)(y % (4*f))/(4*f), (float)(x % (4*f))/(4*f));

                if(x == 79 && y == 0)
                {
                    printf("Serial Window\n");
                    for(int yy = 0; yy < 4; yy++)
                    {
                        for(int xx = 0; xx < 4; xx++)
                        {
                            printf("[(%3.3f,%3.3f,%3.3f)],\t", window_r[yy][xx], window_g[yy][xx], window_b[yy][xx]);
                        }
                        printf("\n");
                    }
                }

                float temp1 = bicubicInterpolate(window_r, (float)(y % f) / f, (float)(x % f) / f);
                float temp2 = bicubicInterpolate(window_g, (float)(y % f) / f, (float)(x % f) / f);
                float temp3 = bicubicInterpolate(window_b, (float)(y % f) / f, (float)(x % f) / f);

                big_img_data[3 * (y * big_width + x) + 0] = (unsigned char)temp1;
                big_img_data[3 * (y * big_width + x) + 1] = (unsigned char)temp2;
                big_img_data[3 * (y * big_width + x) + 2] = (unsigned char)temp3;
            }
            else
            {
                big_img_data[3 * (y * big_width + x) + 0] = img_data[3 * ((y / f) * width + (x / f)) + 0];
                big_img_data[3 * (y * big_width + x) + 1] = img_data[3 * ((y / f) * width + (x / f)) + 1];
                big_img_data[3 * (y * big_width + x) + 2] = img_data[3 * ((y / f) * width + (x / f)) + 2];
            }
        }
    }
}
