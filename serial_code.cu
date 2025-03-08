#include "serial_code.cuh"

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
                            window_r[l][k] = (float)img_data[3 * ((l + y / scale) * width + x / scale + k) + 0];
                            window_g[l][k] = (float)img_data[3 * ((l + y / scale) * width + x / scale + k) + 1];
                            window_b[l][k] = (float)img_data[3 * ((l + y / scale) * width + x / scale + k) + 2];
                        }
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
