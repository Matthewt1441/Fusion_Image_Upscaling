#include "helper.cuh"
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