#pragma once

//#define USE_SDL

typedef struct __align__(4) { // Or alignas(4) in C++11
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a; // Padding to ensure 4-byte alignment
}RGBA_t;

__global__ void rgbToRGBA_Kernel(RGBA_t* d_RGBA_img, unsigned char* d_rgb_img, int numpixels);
__global__ void rgbaToRGB_Kernel(unsigned char* d_rgb_img, RGBA_t* d_rgba_img, int numpixels);

void Image_Compare(unsigned char* img1, unsigned char* img2, int width, int height);
void Grey_Image_Compare(unsigned char* img1, unsigned char* img2, int width, int height);
float cubicKernel(float x, float a = -0.5);