#include "helper.cuh"
#include <stdlib.h>
#include <cmath>

void Average(float* Avg, unsigned char* img_1, unsigned char* img_2, int* width, int* height)
{
    float average_1[CHN_NUM] = { 0 };
    float average_2[CHN_NUM] = { 0 };

    float total_pix = *height * *width;

    for (int y = 0; y < *height; y++)
    {
        for (int x = 0; x < *width; x++)
        {
            for (int img = 0; img < 2; img++)
            {
                for (int chn = 0; chn < CHN_NUM; chn++)
                {
                    average_1[chn] += img_1[CHN_NUM * (y * *width + x) + chn];
                    average_2[chn] += img_2[CHN_NUM * (y * *width + x) + chn];
                }
            }
        }
    }

    for (int chn = 0; chn < CHN_NUM; chn++)
    {
        Avg[0 + chn] = average_1[chn] / total_pix;
        Avg[CHN_NUM + chn] = average_2[chn] / total_pix;
    }
}

void StandardDeviationSquare(float* STD, float* Avg, unsigned char* img_1, unsigned char* img_2, int* width, int* height)
{
    float STD_1[CHN_NUM] = { 0 };
    float STD_2[CHN_NUM] = { 0 };
    float STD_1_2[CHN_NUM] = { 0 };

    float total_pix = *height * *width;
    int count = 0;

    for (int y = 0; y < *height; y++)
    {
        for (int x = 0; x < *width; x++)
        {
            for (int img = 0; img < 2; img++)
            {
                for (int chn = 0; chn < CHN_NUM; chn++)
                {
                    STD_1[chn] += img_1[CHN_NUM * (y * *width + x) + chn] - Avg[0 + chn];
                    STD_2[chn] += img_2[CHN_NUM * (y * *width + x) + chn] - Avg[CHN_NUM + chn];
                    STD_1_2[chn] += (STD_1[chn] * STD_2[chn]);
                }
            }
        }
    }

    for (int chn = 0; chn < CHN_NUM; chn++)
    {
        STD[CHN_NUM * 0 + chn] = STD_1[chn] / total_pix;    //Sigma 1
        STD[CHN_NUM * 1 + chn] = STD_2[chn] / total_pix;    //Sigma 2
        STD[CHN_NUM * 2 + chn] = STD_1_2[chn] / total_pix;    //Sigma 1,2
    }
}

void SSIM(float* ssim, unsigned char* img_1, unsigned char* img_2, int* width, int* height)
{
    float* img_averages = (float*)malloc(sizeof(float) * CHN_NUM * 2);
    float* img_std = (float*)malloc(sizeof(float) * CHN_NUM * 3);

    float L_2 = 255 * 255;  //Max RGB Value ^ 2

    //Constants
    float K1_2 = 0.01 * 0.01;       float K2_2 = 0.03 * 0.03;
    float C1 = K1_2 * L_2;          float C2 = K2_2 * L_2;

    Average(img_averages, img_1, img_2, width, height);
    StandardDeviationSquare(img_std, img_averages, img_1, img_2, width, height);

    for (int chn = 0; chn < CHN_NUM; chn++)
        ssim[chn] = ((2 * img_averages[chn] * img_averages[CHN_NUM + chn] + C1) * (2 * img_std[CHN_NUM * 3 + chn] + C2)) /
        ((img_averages[chn] * img_averages[chn] + img_averages[CHN_NUM + chn] * img_averages[CHN_NUM + chn] + C1) * (img_std[CHN_NUM * 0 + chn] + img_std[CHN_NUM * 1 + chn] + C2));

}

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

    //For now, generate a smaller image.
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    if (((y+i) * width + (x+j)) < (width * height))
                    {
                        window_img1[i][j] = img_1[(y+i) * width + (x+j)];
                        window_img2[i][j] = img_2[(y+i) * width + (x+j)];
                    }
                    else
                    {
                        window_img1[i][j] = -1;
                        window_img2[i][j] = -1;
                    }
                }
            }

            ssim_map[y * width + x] = calculateSSIM(window_img1, window_img2, 8, 8);

        }
    }

}


void ABS_Difference(unsigned char* img_diff, unsigned char* img_1, unsigned char* img_2, int* width, int* height)   //Overloaded function
{
    for (int y = 0; y < *height; y++)
    {
        for (int x = 0; x < *width; x++)
        {
            img_diff[3 * (y * *width + x) + 0] = abs(img_1[3 * (y * *width + x) + 0] - img_2[3 * (y * *width + x) + 0]);
            img_diff[3 * (y * *width + x) + 1] = abs(img_1[3 * (y * *width + x) + 1] - img_2[3 * (y * *width + x) + 1]);
            img_diff[3 * (y * *width + x) + 2] = abs(img_1[3 * (y * *width + x) + 2] - img_2[3 * (y * *width + x) + 2]);
        }
    }
}

void ABS_Difference_Grey(float* diff_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{

    int img_1_signed = 0;
    int img_2_signed = 0;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            img_1_signed = (int)img_1[y*width + x];
            img_2_signed = (int)img_2[y*width + x];
            diff_map[y*width + x] = (float)abs(img_1_signed - img_2_signed);
        }
    }
}

//FLOAT IMPLEMENTATIONS THESE HAVE SPECIAL CHECKS FOR "NEGATIVE" PIXELS

void Average(float* Avg, float* img_1, float* img_2, int* width, int* height)
{
    float average_1[CHN_NUM] = { 0 };
    float average_2[CHN_NUM] = { 0 };

    float total_pix = 0;
    float pix1, pix2;

    for (int y = 0; y < *height; y++)
    {
        for (int x = 0; x < *width; x++)
        {
            for (int chn = 0; chn < CHN_NUM; chn++)
            {
                pix1 = img_1[CHN_NUM * (y * *width + x) + chn];
                pix2 = img_2[CHN_NUM * (y * *width + x) + chn];

                if (!((pix1 < 0) || (pix2 < 0)))    //Check for missing pixel
                {
                    average_1[chn] += pix1;
                    average_2[chn] += pix2;
                    total_pix += 1;
                }
            }
        }
    }

    for (int chn = 0; chn < CHN_NUM; chn++)
    {
        Avg[0 + chn] = average_1[chn] / total_pix / CHN_NUM;
        Avg[CHN_NUM + chn] = average_2[chn] / total_pix / CHN_NUM;
    }
}

void StandardDeviationSquare(float* STD, float* Avg, float* img_1, float* img_2, int* width, int* height)
{
    float STD_1[CHN_NUM] = { 0 };
    float STD_2[CHN_NUM] = { 0 };
    float STD_1_2[CHN_NUM] = { 0 };

    float total_pix = 0;
    float pix1, pix2;

    for (int y = 0; y < *height; y++)
    {
        for (int x = 0; x < *width; x++)
        {
            for (int chn = 0; chn < CHN_NUM; chn++)
            {
                pix1 = img_1[CHN_NUM * (y * *width + x) + chn];
                pix2 = img_2[CHN_NUM * (y * *width + x) + chn];

                if (!((pix1 < 0) || (pix2 < 0)))    //Check for missing pixel
                {
                    STD_1[chn] += (pix1 - Avg[0 + chn]);
                    STD_2[chn] += (pix2 - Avg[CHN_NUM + chn]);
                    STD_1_2[chn] += (STD_1[chn] * STD_2[chn]);
                    total_pix += 1;
                }
            }
        }
    }

    for (int chn = 0; chn < CHN_NUM; chn++)
    {
        STD[CHN_NUM * 0 + chn] = STD_1[chn] / total_pix;    //Sigma 1
        STD[CHN_NUM * 1 + chn] = STD_2[chn] / total_pix;    //Sigma 2
        STD[CHN_NUM * 2 + chn] = STD_1_2[chn] / total_pix;    //Sigma 1,2
    }
}

void SSIM(float* ssim, float* img_1, float* img_2, int* width, int* height)
{
    float* img_averages = (float*)malloc(sizeof(float) * CHN_NUM * 2);
    float* img_std = (float*)malloc(sizeof(float) * CHN_NUM * 3);

    float L_2 = 255 * 255;  //Max RGB Value ^ 2

    //Constants
    float K1_2 = 0.01 * 0.01;       float K2_2 = 0.03 * 0.03;
    float C1 = K1_2 * L_2;          float C2 = K2_2 * L_2;

    Average(img_averages, img_1, img_2, width, height);
    StandardDeviationSquare(img_std, img_averages, img_1, img_2, width, height);

    for (int chn = 0; chn < CHN_NUM; chn++)
        ssim[chn] = ((2 * img_averages[chn] * img_averages[CHN_NUM + chn] + C1) * (2 * img_std[CHN_NUM * 3 + chn] + C2)) /
        ((img_averages[chn] * img_averages[chn] + img_averages[CHN_NUM + chn] * img_averages[CHN_NUM + chn] + C1) * (img_std[CHN_NUM * 0 + chn] + img_std[CHN_NUM * 1 + chn] + C2));

    //FREE THE AVERAGES & STD
    free(img_averages); free(img_std);
}

void ABS_Difference(float* img_diff, float* img_1, float* img_2, int* width, int* height)
{
    for (int y = 0; y < *height; y++)
    {
        for (int x = 0; x < *width; x++)
        {
            img_diff[3 * (y * *width + x) + 0] = abs(img_1[3 * (y * *width + x) + 0] - img_2[3 * (y * *width + x) + 0]);
            img_diff[3 * (y * *width + x) + 1] = abs(img_1[3 * (y * *width + x) + 1] - img_2[3 * (y * *width + x) + 1]);
            img_diff[3 * (y * *width + x) + 2] = abs(img_1[3 * (y * *width + x) + 2] - img_2[3 * (y * *width + x) + 2]);
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
            product_map[idx] = ((map_1[idx]/255.0) * map_2[idx]);
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