
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <string>

const int CHN_NUM = 3;

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
        Avg[0       + chn]    = average_1[chn] / total_pix;
        Avg[CHN_NUM + chn]    = average_2[chn] / total_pix;
    }
}

void StandardDeviationSquare(float* STD, float* Avg, unsigned char* img_1, unsigned char* img_2, int* width, int* height)
{
    float STD_1[CHN_NUM] = { 0 };
    float STD_2[CHN_NUM] = { 0 };
    float STD_1_2[CHN_NUM] = { 0 };

    float total_pix = *height * *width;

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
                    STD_1_2[chn] += STD_1[chn] + STD_2[chn];
                }
            }
        }
    }
    
    for (int chn = 0; chn < CHN_NUM; chn++)
    {
        STD[CHN_NUM * 0 + chn] = STD_1[chn]     / total_pix;    //Sigma 1
        STD[CHN_NUM * 1 + chn] = STD_2[chn]     / total_pix;    //Sigma 2
        STD[CHN_NUM * 2 + chn] = STD_1_2[chn]   / total_pix;    //Sigma 1,2
    }

    
}

void SSIM(float* ssim, unsigned char* img_1, unsigned char* img_2, int* width, int* height)
{
    float* img_averages = (float*)malloc(sizeof(float) * CHN_NUM * 2);
    float* img_std = (float*)malloc(sizeof(float) * CHN_NUM * 3);
    
    float L_2 = 255 * 255;  //Max RGB Value ^ 2
    
    //Constants
    float K1_2 = 0.01;              float K2_2 = 0.03;
    float C1 = K1_2 * L_2;          float C2 = K2_2 * L_2;

    Average(img_averages, img_1, img_2, width, height);
    StandardDeviationSquare(img_std, img_averages, img_1, img_2, width, height);

    for (int chn = 0; chn < CHN_NUM; chn++)
        ssim[chn] = ((2 * img_averages[chn] * img_averages[CHN_NUM + chn] + C1) * (2 * img_std[CHN_NUM * 3 + chn] + C2)) /
                    ((img_averages[chn] * img_averages[chn] + img_averages[CHN_NUM + chn] * img_averages[CHN_NUM + chn] + C1) * (img_std[CHN_NUM * 0 + chn] + img_std[CHN_NUM * 1 + chn] + C2));

}

void ABS_Difference(unsigned char* img_diff, unsigned char* img_1, unsigned char* img_2, int* width, int* height)
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

void Artifact_Detection(unsigned char* img_1, unsigned char* img_2, int* width, int* height, int window_size)
{
    int num_windows_x = ceil(*width / (float) window_size);
    int num_windows_y = ceil(*height / (float) window_size);

    unsigned char* img1_window = (unsigned char*)malloc(sizeof(unsigned char) * window_size * window_size * 3);
    unsigned char* img2_window = (unsigned char*)malloc(sizeof(unsigned char) * window_size * window_size * 3);

    int img_x = 0; 
    int img_y = 0;

    for (int win_y = 0; win_y < num_windows_y; win_y++)
    {
        for (int win_x = 0; win_x < num_windows_x; win_x++)
        {
            //DATA LOADING PHASE
            for (int y = 0; y < window_size; y++)
            {
                for (int x = 0; x < window_size; x++)
                {
                    img_x = x + win_x * window_size;
                    img_y = y + win_y * window_size;

                    if ((img_y < *height) && (img_x < *width))
                    {
                        img1_window[(y * window_size + x) * CHN_NUM] = img_1[(img_y * *width + img_x) * CHN_NUM];
                        img2_window[(y * window_size + x) * CHN_NUM] = img_2[(img_y * *width + img_x) * CHN_NUM];
                    }
                }
            }

            //Image Difference Phase

        }

    }

    
}

void nearestNeighbors(unsigned char* big_img_data, int* big_width, int* big_height, unsigned char* img_data, int* width, int* height, int scale)
{
    int small_x, small_y;

    for (int y = 0; y < *big_height; y++)
    {
        for (int x = 0; x < *big_width; x++)
        {
            small_x = x / scale;
            small_y = y / scale;

            big_img_data[3 * (y * *big_width + x) + 0] = img_data[3 * (small_y * *width + small_x) + 0];
            big_img_data[3 * (y * *big_width + x) + 1] = img_data[3 * (small_y * *width + small_x) + 1];
            big_img_data[3 * (y * *big_width + x) + 2] = img_data[3 * (small_y * *width + small_x) + 2];
        }
    }
}

float cubicInterpolate(float p[4], float x) {
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

float bicubicInterpolate(float p[4][4], float x, float y) {
    float arr[4];
    arr[0] = cubicInterpolate(p[0], y);
    arr[1] = cubicInterpolate(p[1], y);
    arr[2] = cubicInterpolate(p[2], y);
    arr[3] = cubicInterpolate(p[3], y);
    return cubicInterpolate(arr, x);
}


void bicubicInterpolation(unsigned char* big_img_data, int* big_width, int* big_height, unsigned char* img_data, int* width, int* height, int scale)
{
    float window_r[4][4];
    float window_g[4][4];
    float window_b[4][4];

    int f = scale;
    int w = *width;
    int h = *height;

    for (int y = 0; y < 4; y++)
    {
        for (int x = 0; x < 4; x++)
        {
            window_r[y][x] = 0;
            window_g[y][x] = 0;
            window_b[y][x] = 0;
        }
    }

    for (int y = 0; y < f * h; y++)
    {
        for (int x = 0; x < f * w; x++)
        {
            if ((y / f + 4 < h) && (x / f + 4 < w))
            {
                for (int l = 0; l < 4; l++)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        if ((y / f + l < h) && (x / f + k < w))
                        {
                            window_r[l][k] = (float)img_data[3 * ((l + y / scale) * *width + x / scale + k) + 0];
                            window_g[l][k] = (float)img_data[3 * ((l + y / scale) * *width + x / scale + k) + 1];
                            window_b[l][k] = (float)img_data[3 * ((l + y / scale) * *width + x / scale + k) + 2];
                        }
                    }
                }

                float temp1 = bicubicInterpolate(window_r, (float)(y % f) / f, (float)(x % f) / f);
                float temp2 = bicubicInterpolate(window_g, (float)(y % f) / f, (float)(x % f) / f);
                float temp3 = bicubicInterpolate(window_b, (float)(y % f) / f, (float)(x % f) / f);

                big_img_data[3 * (y * *big_width + x) + 0]  = (unsigned char)temp1;
                big_img_data[3 * (y * *big_width + x) + 1] = (unsigned char)temp2;
                big_img_data[3 * (y * *big_width + x) + 2] = (unsigned char)temp3;
            }
            else
            {
                big_img_data[3 * (y * *big_width + x) + 0] = img_data[3 * ((y / f) * *width + (x / f)) + 0];
                big_img_data[3 * (y * *big_width + x) + 1] = img_data[3 * ((y / f) * *width + (x / f)) + 1];
                big_img_data[3 * (y * *big_width + x) + 2] = img_data[3 * ((y / f) * *width + (x / f)) + 2];
            }
        }
    }
}

char* readPPM(char* filename, int* width, int* height) {
    //std::ifstream file(filename, std::ios::binary);

    std::ifstream file(filename, std::ios::binary); // open the file and throw exception if it doesn't exist
    if (file.fail())
        throw "File failed to open";

    std::string magicNumber;
    int maxColorValue;
    int w = 0;
    int h = 0;

    file >> magicNumber;
    file >> w >> h >> maxColorValue;

    file.get(); // skip the trailing white space

    size_t size = w * h * 3;
    char* pixel_data = new char[size];

    file.read(pixel_data, size);

    *width = w;
    *height = h;

    return pixel_data;
}

void writePPM(char* filename, char* img_data, int* width, int* height)
{
    std::ofstream file(filename, std::ios::binary);
    if (file.fail())
        throw "File failed to open";

    file << "P6" << "\n" << *width << " " << *height << "\n" << 255 << "\n";

    size_t size = (*width) * (*height) * 3;

    file.write(img_data, size);
}

char* createImage(char* filename, int* width, int* height)
{
    char* img = (char*)malloc(sizeof(char) * *width * *height * 3);
    char pixel;

    for (int y = 0; y < *height; y++)
    {
        pixel = rand() % 256;
        for (int x = 0; x < *width; x++)
        {
            img[3 * (y * *width + x) + 0] = pixel;
            img[3 * (y * *width + x) + 1] = pixel;
            img[3 * (y * *width + x) + 2] = pixel;
        }
    }

    writePPM(filename, img, width, height);
    return img;
}

int main()
{
    try
    {

        int* width = (int*)malloc(sizeof(int));
        int* height = (int*)malloc(sizeof(int));
        unsigned char* img;

        int* big_width = (int*)malloc(sizeof(int));
        int* big_height = (int*)malloc(sizeof(int));
        int scale = 3;

        //*width = 320;
        //*height = 240;
        //img = createImage("test.ppm", width, height);

        img = (unsigned char*)readPPM("Tuna_2.ppm", width, height);

        *big_width = *width * scale; *big_height = *height * scale;
        unsigned char* big_img_nn = (unsigned char*)malloc(sizeof(unsigned char) * *big_width * *big_height * 3);
        unsigned char* big_img_bic = (unsigned char*)malloc(sizeof(unsigned char) * *big_width * *big_height * 3);
        unsigned char* big_img_dif = (unsigned char*)malloc(sizeof(unsigned char) * *big_width * *big_height * 3);

        printf("Image dimensions: %d x %d\n", *width, *height);
        printf("Upscale Image dimensions: %d x %d\n", *big_width, *big_height);

        nearestNeighbors(big_img_nn, big_width, big_height, img, width, height, scale);
        //nearestNeighbors(big_img_bic, big_width, big_height, img, width, height, scale);
        bicubicInterpolation(big_img_bic, big_width, big_height, img, width, height, scale);
        ABS_Difference(big_img_dif, big_img_nn, big_img_bic, big_width, big_height);

        writePPM("output_NN.ppm", (char*)big_img_nn, big_width, big_height);
        writePPM("output_BIC.ppm", (char*)big_img_bic, big_width, big_height);
        writePPM("output_diff.ppm", (char*)big_img_dif, big_width, big_height);

        free(img);      free(big_img_nn);   free(big_img_bic); free(big_img_dif);
        free(width);    free(big_width);
        free(height);   free(big_height);
    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}