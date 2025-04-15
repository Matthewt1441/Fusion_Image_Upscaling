#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <math.h>
//Initial Naive approach
__global__ void rgbToRGBA_Kernel(RGBA_t* d_RGBA_img, unsigned char* d_rgb_img, int numpixels)
{
    // Each thread processes one pixel
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int sharedIdx = tid * 3;

    // Shared memory for the current block of RGB values
    __shared__ unsigned char sharedRGB[256 * 3];  // Assuming block size is 256 threads

    // Load data into shared memory
    if(idx < numpixels)
    {
        sharedRGB[sharedIdx + 0] = d_rgb_img[idx * 3 + 0];
        sharedRGB[sharedIdx + 1] = d_rgb_img[idx * 3 + 1];
        sharedRGB[sharedIdx + 2] = d_rgb_img[idx * 3 + 2];
    }

    // Synchronize to ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Now process the RGB to RGBA conversion in shared memory
    if(idx < numpixels) {
        // Read RGB values from shared memory
        unsigned char r = sharedRGB[sharedIdx + 0];
        unsigned char g = sharedRGB[sharedIdx + 1];
        unsigned char b = sharedRGB[sharedIdx + 2];

        // Write to RGBA array (global memory)
        d_RGBA_img[idx].r = r;
        d_RGBA_img[idx].g = g;
        d_RGBA_img[idx].b = b;
        d_RGBA_img[idx].a = 255;  // Alpha is fully opaque
    }
}


//Initial Naive approach
__global__ void rgbaToRGB_Kernel(unsigned char* d_rgb_img, RGBA_t* d_rgba_img, int numpixels)
{
    // Each thread processes one pixel
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int sharedIdx = threadIdx.x;

    // Shared memory for the current block of RGBA values
    __shared__ RGBA_t sharedRGB[256];  // Assuming block size is 256 threads

    // Load data into shared memory
    if(idx < numpixels)
    {
        sharedRGB[sharedIdx] = d_rgba_img[idx];
    }

    // Synchronize to ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Now process the RGBA to RGB conversion in shared memory
    if(idx < numpixels) {
        // Read RGBA values from shared memory
        RGBA_t rgba_val = sharedRGB[sharedIdx];

        // Write to RGB array (global memory)
        d_rgb_img[idx * 3 + 0] = rgba_val.r;
        d_rgb_img[idx * 3 + 1] = rgba_val.g;
        d_rgb_img[idx * 3 + 2] = rgba_val.b;
    }
}

void Image_Compare(unsigned char* img1, unsigned char* img2, int width, int height)
{
    int idx = 0;
    int y;
    int x;
    bool pass = true;
    for( y = 0; y < height; y++) 
    {
        for(x = 0; x < width; x++)
        {
            idx = y * width + x;

            //                 R                                   G                                   B
            if((img1[idx + 0] != img2[idx + 0]) || (img1[idx + 1] != img2[idx + 1]) || (img1[idx + 2] != img2[idx + 2]))
            {
                pass = false;
                goto LOOP_EXIT;
            }


        }
    }

LOOP_EXIT:
    if(!pass)
    {
        printf("Images do not match at pixel X: %d, Y: %d\n", x, y);
    }
    else
    {
        printf("Images match!\n");
    }

}

void Grey_Image_Compare(unsigned char* img1, unsigned char* img2, int width, int height)
{
    int idx = 0;
    int y;
    int x;
    bool pass = true;
    for( y = 0; y < height; y++) 
    {
        for(x = 0; x < width; x++)
        {
            idx = y * width + x;
           
            if(img1[idx] != img2[idx])
            {
                pass = false;
                goto GREY_LOOP_EXIT;
            }


        }
    }

GREY_LOOP_EXIT:
    if(!pass)
    {
        printf("Images do not match at pixel X: %d, Y: %d\n", x, y);
    }
    else
    {
        printf("Images match!\n");
    }

}


float cubicKernel(float x, float a)
{
     x = abs(x);
     if(x <= 1.0f)
     {
         return (a + 2) * x * x * x - (a + 3) * x * x + 1;
     }
     else if(x < 2.0f)
     {
         return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a;
     }
     else
     {
         return 0.0f;
     }
}