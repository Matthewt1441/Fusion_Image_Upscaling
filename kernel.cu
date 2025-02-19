
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <string>

void nearestNeighbors(unsigned char* big_img_data, int* big_width, int* big_height, unsigned char* img_data, int* width, int* height, int scale)
{
    int small_x, small_y;

    for (int y = 0; y < *big_height; y++)
        for (int x = 0; x < *big_width; x++)
        {
            small_x = x / scale;
            small_y = y / scale;

            big_img_data[3 * (y * *big_width + x) + 0] = img_data[3 * (small_y * *width + small_x) + 0];
            big_img_data[3 * (y * *big_width + x) + 1] = img_data[3 * (small_y * *width + small_x) + 1];
            big_img_data[3 * (y * *big_width + x) + 2] = img_data[3 * (small_y * *width + small_x) + 2];
        }
}

//float bicubicpol(float x, float y, float p[4][4]) {
//
//    float a00, a01, a02, a03;
//    float a10, a11, a12, a13;
//    float a20, a21, a22, a23;
//    float a30, a31, a32, a33;
//    float x2 = x * x;
//    float x3 = x2 * x;
//    float y2 = y * y;
//    float y3 = y2 * y;
//
//    a00 = p[1][1];
//    a01 = -.5 * p[1][0] + .5 * p[1][2];
//    a02 = p[1][0] - 2.5 * p[1][1] + 2 * p[1][2] - .5 * p[1][3];
//    a03 = -.5 * p[1][0] + 1.5 * p[1][1] - 1.5 * p[1][2] + .5 * p[1][3];
//    a10 = -.5 * p[0][1] + .5 * p[2][1];
//    a11 = .25 * p[0][0] - .25 * p[0][2] - .25 * p[2][0] + .25 * p[2][2];
//    a12 = -.5 * p[0][0] + 1.25 * p[0][1] - p[0][2] + .25 * p[0][3] + .5 * p[2][0] - 1.25 * p[2][1] + p[2][2] - .25 * p[2][3];
//    a13 = .25 * p[0][0] - .75 * p[0][1] + .75 * p[0][2] - .25 * p[0][3] - .25 * p[2][0] + .75 * p[2][1] - .75 * p[2][2] + .25 * p[2][3];
//    a20 = p[0][1] - 2.5 * p[1][1] + 2 * p[2][1] - .5 * p[3][1];
//    a21 = -.5 * p[0][0] + .5 * p[0][2] + 1.25 * p[1][0] - 1.25 * p[1][2] - p[2][0] + p[2][2] + .25 * p[3][0] - .25 * p[3][2];
//    a22 = p[0][0] - 2.5 * p[0][1] + 2 * p[0][2] - .5 * p[0][3] - 2.5 * p[1][0] + 6.25 * p[1][1] - 5 * p[1][2] + 1.25 * p[1][3] + 2 * p[2][0] - 5 * p[2][1] + 4 * p[2][2] - p[2][3] - .5 * p[3][0] + 1.25 * p[3][1] - p[3][2] + .25 * p[3][3];
//    a23 = -.5 * p[0][0] + 1.5 * p[0][1] - 1.5 * p[0][2] + .5 * p[0][3] + 1.25 * p[1][0] - 3.75 * p[1][1] + 3.75 * p[1][2] - 1.25 * p[1][3] - p[2][0] + 3 * p[2][1] - 3 * p[2][2] + p[2][3] + .25 * p[3][0] - .75 * p[3][1] + .75 * p[3][2] - .25 * p[3][3];
//    a30 = -.5 * p[0][1] + 1.5 * p[1][1] - 1.5 * p[2][1] + .5 * p[3][1];
//    a31 = .25 * p[0][0] - .25 * p[0][2] - .75 * p[1][0] + .75 * p[1][2] + .75 * p[2][0] - .75 * p[2][2] - .25 * p[3][0] + .25 * p[3][2];
//    a32 = -.5 * p[0][0] + 1.25 * p[0][1] - p[0][2] + .25 * p[0][3] + 1.5 * p[1][0] - 3.75 * p[1][1] + 3 * p[1][2] - .75 * p[1][3] - 1.5 * p[2][0] + 3.75 * p[2][1] - 3 * p[2][2] + .75 * p[2][3] + .5 * p[3][0] - 1.25 * p[3][1] + p[3][2] - .25 * p[3][3];
//    a33 = .25 * p[0][0] - .75 * p[0][1] + .75 * p[0][2] - .25 * p[0][3] - .75 * p[1][0] + 2.25 * p[1][1] - 2.25 * p[1][2] + .75 * p[1][3] + .75 * p[2][0] - 2.25 * p[2][1] + 2.25 * p[2][2] - .75 * p[2][3] - .25 * p[3][0] + .75 * p[3][1] - .75 * p[3][2] + .25 * p[3][3];
//
//
//    return (a00 + a01 * y + a02 * y2 + a03 * y3) +
//        (a10 + a11 * y + a12 * y2 + a13 * y3) * x +
//        (a20 + a21 * y + a22 * y2 + a23 * y3) * x2 +
//        (a30 + a31 * y + a32 * y2 + a33 * y3) * x3;
//
//}


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
                            //window_r[l][k] = img_data[3 * ((y / f) * *width + x / f) + 0];
                            //window_g[l][k] = img_data[3 * ((y / f) * *width + x / f) + 1];
                            //window_b[l][k] = img_data[3 * ((y / f) * *width + x / f) + 2];
                        }
                    }
                }

                float temp1 = bicubicInterpolate(window_r, (float)(y % f) / f, (float)(x % f) / f);
                float temp2 = bicubicInterpolate(window_g, (float)(y % f) / f, (float)(x % f) / f);
                float temp3 = bicubicInterpolate(window_b, (float)(y % f) / f, (float)(x % f) / f);

                big_img_data[3 * (y * *big_width + x) + 0] = (unsigned char)temp1;
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

        printf("Image dimensions: %d x %d\n", *width, *height);
        printf("Upscale Image dimensions: %d x %d\n", *big_width, *big_height);

        nearestNeighbors(big_img_nn, big_width, big_height, img, width, height, scale);
        //nearestNeighbors(big_img_bic, big_width, big_height, img, width, height, scale);
        bicubicInterpolation(big_img_bic, big_width, big_height, img, width, height, scale);

        writePPM("output_NN.ppm", (char*)big_img_nn, big_width, big_height);
        writePPM("output_BIC.ppm", (char*)big_img_bic, big_width, big_height);

        free(img);      free(big_img_nn);   free(big_img_bic);
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