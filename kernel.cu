#define _USE_MATH_DEFINES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "util.cu"

#include <chrono>

#include <chrono>

#include <SDL.h>
#undef main
#include <SDL_ttf.h>
#undef main


typedef struct __align__(4) { // Or alignas(4) in C++11
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a; // Padding to ensure 4-byte alignment
}RGBA_t;

char* readPPM(char* filename, int* width, int* height)
{
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

char* readPPM(char* pixel_data, char* filename, int* width, int* height)
{
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

    file.read(pixel_data, size);

    *width = w;
    *height = h;

    return pixel_data;
}

char* readPPMGray(char* filename, int* width, int* height)
{
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

    size_t size = w * h;
    char* pixel_data = new char[size];

    file.read(pixel_data, size);

    *width = w;
    *height = h;

    return pixel_data;
}

void writePPM(char* filename, char* img_data, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (file.fail())
        throw "File failed to open";

    file << "P6" << "\n" << width << " " << height << "\n" << 255 << "\n";

    size_t size = (width) * (height) * 3;

    file.write(img_data, size);
}

void writePPMGrey(char* filename, char* img_data, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (file.fail())
        throw "File failed to open";

    file << "P5" << "\n" << width << " " << height << "\n" << 255 << "\n";

    size_t size = (width) * (height);

    file.write(img_data, size);
}

void Image_Compare(unsigned char* img1, unsigned char* img2, int width, int height)
{
    int idx = 0;
    int y;
    int x;
    bool pass = true;
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            idx = y * width + x;
            char img1_r = img1[idx + 0];
            char img1_g = img1[idx + 1];
            char img1_b = img1[idx + 2];
            char img2_r = img2[idx + 0];
            char img2_g = img2[idx + 1];
            char img2_b = img2[idx + 2];

            if ((img2_r < img1_r - 5) || (img2_r > img1_r + 5))
            {
                pass = false;
                goto LOOP_EXIT;
            }

            if ((img2_g < img1_g - 5) || (img2_g > img1_g + 5))
            {
                pass = false;
                goto LOOP_EXIT;
            }

            if ((img2_b < img1_b - 5) || (img2_b > img1_b + 5))
            {
                pass = false;
                goto LOOP_EXIT;
            }

        }
    }

LOOP_EXIT:
    if (!pass)
    {
        printf("Images do not match at pixel X: %d, Y: %d, Img1 [%d, %d, %d], Img2 [%d, %d, %d]\n", x, y, img1[idx + 0], img1[idx + 1], img1[idx + 2], img2[idx + 0], img2[idx + 1], img2[idx + 2]);

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
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            idx = y * width + x;

            if (img1[idx] < img2[idx] - 5 || img1[idx] > img2[idx] + 5)
            {
                pass = false;
                goto GREY_LOOP_EXIT;
            }


        }
    }

GREY_LOOP_EXIT:
    if (!pass)
    {
        printf("Images do not match at pixel X: %d, Y: %d, Img1 [%d, %d, %d], Img2 [%d, %d, %d]\n", x, y, img1[idx + 0], img1[idx + 1], img1[idx + 2], img2[idx + 0], img2[idx + 1], img2[idx + 2]);

    }
    else
    {
        printf("Images match!\n");
    }
}


__global__ void RGB2GreyscaleKernel(unsigned char* rgb_img, unsigned char* grey_img, int width, int height)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < height && Col < width)
    {
        int rgbidx = rgbidx = 3 * (Row * width + Col);
        grey_img[Row * width + Col] = (21 * rgb_img[rgbidx + 0] / 100) + (71 * rgb_img[rgbidx + 1] / 100) + (7 * rgb_img[rgbidx + 2] / 100);
    }
}


//Initial Naive approach
__global__ void rgbToRGBA_Kernel(RGBA_t* d_RGBA_img, unsigned char* d_rgb_img, int numpixels)
{
    // Each thread processes one pixel
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int sharedIdx = tid * 3;

    // Shared memory for the current block of RGB values
    extern __shared__ unsigned char sharedRGB_char[];

    // Load data into shared memory
    if (idx < numpixels)
    {
        sharedRGB_char[sharedIdx + 0] = d_rgb_img[idx * 3 + 0];
        sharedRGB_char[sharedIdx + 1] = d_rgb_img[idx * 3 + 1];
        sharedRGB_char[sharedIdx + 2] = d_rgb_img[idx * 3 + 2];
    }

    // Synchronize to ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Now process the RGB to RGBA conversion in shared memory
    if (idx < numpixels) {
        // Read RGB values from shared memory
        unsigned char r = sharedRGB_char[sharedIdx + 0];
        unsigned char g = sharedRGB_char[sharedIdx + 1];
        unsigned char b = sharedRGB_char[sharedIdx + 2];

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
    extern __shared__ RGBA_t sharedRGB[];  // Assuming block size is 256 threads

    // Load data into shared memory
    if (idx < numpixels)
    {
        sharedRGB[sharedIdx] = d_rgba_img[idx];
    }

    // Synchronize to ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Now process the RGBA to RGB conversion in shared memory
    if (idx < numpixels) {
        // Read RGBA values from shared memory
        RGBA_t rgba_val = sharedRGB[sharedIdx];

        // Write to RGB array (global memory)
        d_rgb_img[idx * 3 + 0] = rgba_val.r;
        d_rgb_img[idx * 3 + 1] = rgba_val.g;
        d_rgb_img[idx * 3 + 2] = rgba_val.b;
    }
}

__device__ float bicubicInterpolateDevice_Shared(float p[4][4], float y, float x)
{
    float arr[4];
    float temp;
    float dx_half = 0.5 * x;

    temp = p[0][1] + dx_half * (p[0][2] - p[0][0] + x * (2.0 * p[0][0] - 5.0 * p[0][1] + 4.0 * p[0][2] - p[0][3] + x * (3.0 * (p[0][1] - p[0][2]) + p[0][3] - p[0][0])));
    arr[0] = temp;

    temp = p[1][1] + dx_half * (p[1][2] - p[1][0] + x * (2.0 * p[1][0] - 5.0 * p[1][1] + 4.0 * p[1][2] - p[1][3] + x * (3.0 * (p[1][1] - p[1][2]) + p[1][3] - p[1][0])));
    arr[1] = temp;

    temp = p[2][1] + dx_half * (p[2][2] - p[2][0] + x * (2.0 * p[2][0] - 5.0 * p[2][1] + 4.0 * p[2][2] - p[2][3] + x * (3.0 * (p[2][1] - p[2][2]) + p[2][3] - p[2][0])));
    arr[2] = temp;

    temp = p[3][1] + dx_half * (p[3][2] - p[3][0] + x * (2.0 * p[3][0] - 5.0 * p[3][1] + 4.0 * p[3][2] - p[3][3] + x * (3.0 * (p[3][1] - p[3][2]) + p[3][3] - p[3][0])));
    arr[3] = temp;

    temp = arr[1] + 0.5 * y * (arr[2] - arr[0] + y * (2.0 * arr[0] - 5.0 * arr[1] + 4.0 * arr[2] - arr[3] + y * (3.0 * (arr[1] - arr[2]) + arr[3] - arr[0])));

    temp = temp * ((temp < 256.0) && (temp > -1.0)) + 255 * (temp > 255.0);//+ 0 * (temp < 0);
    return temp;
}

//Run with block sizes that are multiples of the scale
__global__ void bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int g_input_x = 0;
    int g_input_y = 0;

    int g_output_x = Col;
    int g_output_y = Row;

    int tile_input_x = 0;
    int tile_input_y = 0;

    int window_x = 0;
    int window_y = 0;

    //Only based on Block Size
    int tile_width = (blockDim.x / scale) + 3;
    int tile_height = (blockDim.y / scale) + 3;
    extern __shared__ RGBA_t s_tile[];

    float window_r[4][4];
    float window_g[4][4];
    float window_b[4][4];

    RGBA_t rgba_val;

    if (threadIdx.x < tile_width && threadIdx.y < tile_height)
    {
        //Calculate Global Input Index
        g_input_x = blockIdx.x * (blockDim.x / scale) + threadIdx.x - 1;
        g_input_y = blockIdx.y * (blockDim.y / scale) + threadIdx.y - 1;

        // Fill window with Nearest Neighbor edge behavior
        if (g_input_x < 0 || g_input_x >= width)
        {
            // Find nearest in-bounds pixel
            g_input_x = (g_input_x < 0) ? 0 : width - 1;
        }
        // Fill window with Nearest Neighbor edge behavior
        if (g_input_y < 0 || g_input_y >= height)
        {
            // Find nearest in-bounds pixel
            g_input_y = (g_input_y < 0) ? 0 : height - 1;
        }

        s_tile[threadIdx.y * tile_width + threadIdx.x] = img_data[g_input_y * width + g_input_x];
    }
    __syncthreads();

    if (g_output_y < big_height && g_output_x < big_width)
    {
        //Calculate starting index for windows (funky stuff to remove shift)
        float interpolated_x = (((float)threadIdx.x + 0.5f) / (float)scale - 0.5f);
        float interpolated_y = (((float)threadIdx.y + 0.5f) / (float)scale - 0.5f);

        //Round down to nearest index
        int interpolated_idx_x = interpolated_x;
        int interpolated_idx_y = interpolated_y;

        float dx = interpolated_x - interpolated_idx_x;
        float dy = interpolated_y - interpolated_idx_y;

        //Fill local window with tiled input data
        for (window_y = -1; window_y < 3; window_y++)
        {
            for (window_x = -1; window_x < 3; window_x++)
            {
                //Calculate Input Image Tile index
                tile_input_x = interpolated_idx_x + window_x + 1;
                tile_input_y = interpolated_idx_y + window_y + 1;

                rgba_val = s_tile[tile_input_y * tile_width + tile_input_x];

                window_r[window_y + 1][window_x + 1] = (float)rgba_val.r;    //R
                window_g[window_y + 1][window_x + 1] = (float)rgba_val.g;    //G
                window_b[window_y + 1][window_x + 1] = (float)rgba_val.b;    //B
            }
        }

        rgba_val.r = (unsigned char)bicubicInterpolateDevice_Shared(window_r, dy, dx);
        rgba_val.g = (unsigned char)bicubicInterpolateDevice_Shared(window_g, dy, dx);
        rgba_val.b = (unsigned char)bicubicInterpolateDevice_Shared(window_b, dy, dx);

        big_img_data[g_output_y * big_width + g_output_x] = rgba_val;

        grey_big_img_data[g_output_y * big_width + g_output_x] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;

    }

}

__global__ void nearestNeighbors_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int small_x = 0;    int small_y = 0;

    RGBA_t rgba_val;

    if (Row < big_height && Col < big_width)
    {
        small_x = Col / scale;
        small_y = Row / scale;

        rgba_val = img_data[small_y * width + small_x];

        big_img_data[Row * big_width + Col] = rgba_val;

        grey_big_img_data[Row * big_width + Col] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;
    }
}

#define WINDOW_SIZE     8
#define WINDOW_PIXELS   64
__global__ void Artifact_Shared_Memory_Kernel(float* artifact_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    //int window_size = 8;
    //Window size dictates the size of structures that we can detect. Maybe should look into what effect this has
    //on overall image quality & performance
    // Consider the guassian option with an 11x11 window

    extern __shared__ float window_img[];

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, sum12 = 0;
    float img_diff;

    int valid_count = 0;

    //For now, generate a smaller image.

    if (Row < height && Col < width)
    {
        sum1 = img_1[Row * width + Col];    //Using these as temp registers. Just pretend they are called temp1 & temp2
        sum2 = img_2[Row * width + Col];
        window_img[tid_y * WINDOW_SIZE + tid_x + WINDOW_PIXELS] = sum1;
        window_img[tid_y * WINDOW_SIZE + tid_x] = sum2;
        img_diff = (float)abs((sum1 - sum2) / 255.0);
    }

    else
    {
        window_img[tid_y * WINDOW_SIZE + tid_x + WINDOW_PIXELS] = -1;
        window_img[tid_y * WINDOW_SIZE + tid_x] = -1;
    }

    sum1 = 0; sum2 = 0; //reset registers

    __syncthreads();

    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            if ((window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS] >= 0) && (window_img[i * WINDOW_SIZE + j] >= 0))
            {
                sum1 += window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS];
                sum2 += window_img[i * WINDOW_SIZE + j];
                sum1Sq += window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS] * window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS];
                sum2Sq += window_img[i * WINDOW_SIZE + j] * window_img[i * WINDOW_SIZE + j];
                sum12 += window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS] * window_img[i * WINDOW_SIZE + j];
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

    artifact_map[Row * width + Col] = ssim * img_diff;
}

__constant__ float d_guas_kernel_seperable[7] = { 0.0366328470,   0.111280762,    0.216745317,    0.270682156,    0.216745317,    0.111280762,    0.0366328470 };

__global__ void horizontalGuassianBlurConvolve(float* blur_map, float* input_map, int width, int height, int ksize)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    //Shared Memory based on block size and kernel size (always 7 in our case)
    // Tile Width   = Block_Width + KSize - 1
    // Tile Height  = Block_Height
    extern __shared__ float s_tile_h[];

    int tile_width = blockDim.x + ksize - 1;
    int tile_height = blockDim.y;
    int radius = ksize / 2;


    //Fill Input Tile by striding through the input array
    int tile_input_idx_x = tidx;
    int tile_input_idx_y = tidy;
    while (tile_input_idx_x < tile_width)
    {
        int g_input_idx_x = (tile_input_idx_x - radius) + blockIdx.x * blockDim.x;

        if (g_input_idx_x >= 0 && g_input_idx_x < width)
        {
            s_tile_h[tile_input_idx_y * tile_width + tile_input_idx_x] = input_map[Row * width + g_input_idx_x];
        }
        else
        {
            s_tile_h[tile_input_idx_y * tile_width + tile_input_idx_x] = 0;
        }

        //Stride by blockDim ammount
        tile_input_idx_x += blockDim.x;
    }
    __syncthreads();


    float sum = 0;
    if (Row < height && Col < width)
    {
        //Define Starting points for input tile
        int tile_y = tidy;
        int tile_x = tidx + radius;

        //Horizontal Convolve
        for (int k = -radius; k <= radius; k++)
        {
            sum += s_tile_h[tile_y * tile_width + (tile_x - k)] * d_guas_kernel_seperable[k + radius];
        }

        //Global Write
        blur_map[Row * width + Col] = sum;
    }
}

__global__ void verticalGuassianBlurConvolve(float* blur_map, float* input_map, int width, int height, float threshold, int ksize)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    //Shared Memory based on block size and kernel size (always 7 in our case)
    // Tile Width   = Block_Width
    // Tile Height  = Block_Height + KSize - 1
    extern __shared__ float s_tile_v[];

    int tile_width = blockDim.x;
    int tile_height = blockDim.y + ksize - 1;
    int radius = ksize / 2;

    //Fill Input Tile by striding through the input array
    int tile_input_idx_x = tidx;
    int tile_input_idx_y = tidy;
    while (tile_input_idx_y < tile_height)
    {
        int g_input_idx_y = (tile_input_idx_y - radius) + blockIdx.y * blockDim.y;

        if (g_input_idx_y >= 0 && g_input_idx_y < height)
        {
            s_tile_v[tile_input_idx_y * tile_width + tile_input_idx_x] = input_map[g_input_idx_y * width + Col];
        }
        else
        {
            s_tile_v[tile_input_idx_y * tile_width + tile_input_idx_x] = 0;
        }

        //Stride by blockDim ammount
        tile_input_idx_y += blockDim.y;
    }
    __syncthreads();


    float sum = 0;
    if (Row < height && Col < width)
    {
        //Define Starting points for input tile
        int tile_y = tidy + radius;
        int tile_x = tidx;

        //Horizontal Convolve
        for (int k = -radius; k <= radius; k++)
        {
            sum += s_tile_v[(tile_y - k) * tile_width + tile_x] * d_guas_kernel_seperable[k + radius];
        }

        //Global Write
        blur_map[Row * width + Col] = (sum > threshold) ? 1.0 : 0.0;//sum;
    }
}

__global__ void GuassianBlur_Threshold_Map_Naive_Kernel(float* blur_map, float* input_map, int width, int height, int radius, float sigma, float threshold)
{
    //Generate Normalized Guassian Kernal for blurring. This may need to be adjusted so I'll make it flexible.
    //We can eventually hardcode this when we settle on ideal blur.
    int kernel_size = 2 * radius + 1;
    int kernel_center = kernel_size / 2;
    float sum = 0.0;
    float guassian_kernel[49] = { 0 };

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    float my_PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;

    if (Row < height && Col < width)
    {
        for (int y = 0; y < kernel_size; y++)
        {
            for (int x = 0; x < kernel_size; x++)
            {
                double exponent = -((x - kernel_center) * (x - kernel_center) - (y - kernel_center) * (y - kernel_center)) / (2 * sigma * sigma);
                guassian_kernel[y * kernel_size + x] = exp(exponent) / (2 * my_PI * sigma * sigma);
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

__global__ void Image_Fusion_Kernel_RGBA(RGBA_t* fused_img, RGBA_t* img_1, RGBA_t* img_2, float* weight_map, int width, int height)
{
    //int Row = blockIdx.y * blockDim.y + threadIdx.y;
    //int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //int map_idx = Row * width + Col;
    //int img_idx = 3 * map_idx;
    RGBA_t rgba_pxl1;
    RGBA_t rgba_pxl2;
    RGBA_t rgba_fused;

    //if (Row < height && Col < width)
    if (idx < width * height)
    {
        rgba_pxl1 = img_1[idx];
        rgba_pxl2 = img_2[idx];

        rgba_fused.r = rgba_pxl1.r * weight_map[idx] + rgba_pxl2.r * (1.0 - weight_map[idx]);
        rgba_fused.g = rgba_pxl1.g * weight_map[idx] + rgba_pxl2.g * (1.0 - weight_map[idx]);
        rgba_fused.b = rgba_pxl1.b * weight_map[idx] + rgba_pxl2.b * (1.0 - weight_map[idx]);

        fused_img[idx] = rgba_fused;
    }
}

int main()
{
    int width;
    int height;

    int big_width;
    int big_height;
    int big_pixel_count;

    float diff;

    const int STREAM_COUNT = 3;

    unsigned char* h_img[STREAM_COUNT];                            //Original Small Input Image
    unsigned char* h_big_img_fused[STREAM_COUNT];                  //Upscaled Fused Image

    //Device Array Pointers
    unsigned char* d_img[STREAM_COUNT];                            //Original Small Input Image
    RGBA_t* d_RGBA_img[STREAM_COUNT];                       //Original Small Input Image w/ 32bit-pixel format
    RGBA_t* d_big_img_nn[STREAM_COUNT];                     //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    RGBA_t* d_big_img_bic[STREAM_COUNT];                    //Upscaled Bicubic Image w/ 32bit-pixel format
    unsigned char* d_big_img_nn_grey[STREAM_COUNT];                //Upscaled Greyscale Nearest Neighbor Image
    unsigned char* d_big_img_bic_grey[STREAM_COUNT];               //Upscaled Greyscale Bicubic Image
    float* d_big_artifact_map[STREAM_COUNT];               //Upscaled Artifact Map for image fusion
    float* d_big_blurred_artifact_map[STREAM_COUNT];       //Upscaled Blurred Artifact Map for image fusion
    float* d_big_blurred_artifact_map_inter[STREAM_COUNT]; //Upscaled Blurred Artifact Map for image fusion
    RGBA_t* d_big_rgba_img_fused[STREAM_COUNT];             //Upscaled Fused Image w/ 32bit-pixel format
    unsigned char* d_big_img_fused[STREAM_COUNT];                  //Upscaled Fused Image  

    cudaStream_t stream[STREAM_COUNT];

    int block_dim = 16; //The x and y axis size for the block is 16 threads. Total 256 threads
    int window_size = 8;
    int scale = 2;

    bool RUNNING = true;
    bool firstImg = true;
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;
    SDL_Event event;
    SDL_PollEvent(&event);

    try
    {
        unsigned char* h_img_dim = (unsigned char*)readPPM("./LM_Frame/image1.ppm", &width, &height);
        free(h_img_dim);

        //Define big image width and height
        big_width = width * scale; big_height = height * scale;
        big_pixel_count = big_width * big_height;
        int pixel_count = width * height;

        cudaError_t cudaStatus;

        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        }

        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            printf("SDL initialization failed: %c\n", SDL_GetError());
            return 1;
        }

        // Initialize SDL_ttf
        if (TTF_Init() < 0) {
            printf("SDL_ttf could not initialize! TTF_Error: %s\n", TTF_GetError());
            SDL_Quit();
            return EXIT_FAILURE;
        }

        TTF_Font* Sans = TTF_OpenFont("Sans.ttf", 24);
        SDL_Color White = { 255, 255, 255 };

        char fps_str[50];
        char file_name[50];

        // as TTF_RenderText_Solid could only be used on
        // SDL_Surface then you have to create the surface first
        SDL_Surface* fps_msg;
        SDL_Texture* fps_txt;

        SDL_Rect Message_rect; //create a rect
        Message_rect.x = 5;  //controls the rect's x coordinate 
        Message_rect.y = 5; // controls the rect's y coordinte
        Message_rect.w = 200; // controls the width of the rect
        Message_rect.h = 30; // controls the height of the rect
        int count = 0;

        double frame_cap = 10;
        sprintf(fps_str, "FPS:%.*f", 3, 0.0);

        int max_image = 200;
        int current_img = 1;

        double processing_time = 0;


        for (int s = 0; s < STREAM_COUNT; s++)
        {
            cudaStreamCreate(&stream[s]);

            //******** Malloc Host Images ********//
            cudaHostAlloc((void**)&h_img[s], sizeof(unsigned char) * pixel_count * 3, cudaHostAllocDefault);
            cudaHostAlloc((void**)&h_big_img_fused[s], sizeof(unsigned char) * big_pixel_count * 3, cudaHostAllocDefault);
            //******** Malloc Host Images ********//

            //******** Malloc Device Images ********//
            //Original Image & RGBA Image
            if (cudaMalloc((void**)&d_img[s], width * height * sizeof(unsigned char) * 3) != cudaSuccess)
                fprintf(stderr, "Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&d_RGBA_img[s], width * height * sizeof(RGBA_t)) != cudaSuccess)
                fprintf(stderr, "RGBA Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            //Upscaled Images
            if (cudaMalloc((void**)&d_big_img_nn[s], big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
                fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&d_big_img_bic[s], big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
                fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            //Grey Versions for Upscaled Images
            if (cudaMalloc((void**)&d_big_img_nn_grey[s], big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
                fprintf(stderr, "NN Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&d_big_img_bic_grey[s], big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
                fprintf(stderr, "BIC Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            //Maps for Fusion
            if (cudaMalloc((void**)&d_big_artifact_map[s], big_width * big_height * sizeof(float)) != cudaSuccess)
                fprintf(stderr, "Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&d_big_blurred_artifact_map_inter[s], big_width * big_height * sizeof(float)) != cudaSuccess)
                fprintf(stderr, "Intermediate Blured Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&d_big_blurred_artifact_map[s], big_width * big_height * sizeof(float)) != cudaSuccess)
                fprintf(stderr, "Blured Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            //Final Image and RGBA Image
            if (cudaMalloc((void**)&d_big_img_fused[s], big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
                fprintf(stderr, "Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&d_big_rgba_img_fused[s], big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
                fprintf(stderr, "RGBA Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        }

        dim3 RGB_Block(256);
        dim3 RGB_Grid(ceil((big_width * big_height) / (float)RGB_Block.x));
        int  rgbToRGBA_Shared_Mem_Size = sizeof(unsigned char) * RGB_Block.x * 3;
        int  rgbaToRGB_Shared_Mem_Size = sizeof(RGBA_t) * RGB_Block.x;

        dim3 NN_Block(16, 16);
        dim3 NN_Grid(((big_width - 1) / NN_Block.x) + 1, ((big_height - 1) / NN_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double

        dim3 BiCubic_Block(4 * scale, 4 * scale);
        dim3 BiCubic_Grid(((big_width - 1) / BiCubic_Block.x) + 1, ((big_height - 1) / BiCubic_Block.y) + 1);
        int  BiCubic_Shared_Mem_Size = sizeof(RGBA_t) * ((BiCubic_Block.y / scale) + 3) * ((BiCubic_Block.x / scale) + 3);

        dim3 Arti_Block(8, 8);
        dim3 Arti_Grid(((big_width - 1) / Arti_Block.x) + 1, ((big_height - 1) / Arti_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        int  Arti_Shared_Mem_Size = sizeof(float) * 2 * 8 * 8;

        //Setup Guassian Blur based on passed in Args
        int GUAS_Ksize = 7;
        float GUAS_Sigma = 1.5;
        dim3 h_Guas_Block(256, 1);
        dim3 h_Guas_Grid(((big_width - 1) / h_Guas_Block.x) + 1, ((big_height - 1) / h_Guas_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 v_Guas_Block(8, 32);
        dim3 v_Guas_Grid(((big_width - 1) / v_Guas_Block.x) + 1, ((big_height - 1) / v_Guas_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        int  h_Gauss_Mem = sizeof(float) * (h_Guas_Block.x + GUAS_Ksize - 1) * h_Guas_Block.y;
        int  v_Gauss_Mem = sizeof(float) * (v_Guas_Block.y + GUAS_Ksize - 1) * v_Guas_Block.x;


        while (RUNNING && event.type != SDL_QUIT)
        {
            if (count == frame_cap)
            {
                diff = 1000 * frame_cap / processing_time * STREAM_COUNT;
                sprintf(fps_str, "FPS:%.*f", 3, diff);

                count = 0;
                processing_time = 0;
            }

           
            auto start = std::chrono::high_resolution_clock::now();

            for (int s = 0; s < STREAM_COUNT; s++)
            {
                //PHASE 0 : Load Input Image

                sprintf(file_name, "./LM_Frame/image%d.ppm", current_img + s);

                h_img[s] = (unsigned char*)readPPM(file_name, &width, &height);

                //Copy Input Image to Device
                cudaMemcpyAsync(d_img[s], h_img[s], sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice, stream[s]);

                //PHASE 1 : Image Pre-Processing : Convert original image to RGBA image
                rgbToRGBA_Kernel << < RGB_Grid, RGB_Block, rgbToRGBA_Shared_Mem_Size, stream[s] >> > (d_RGBA_img[s], d_img[s], width * height);

                //PHASE 2 : Image Scaling
                nearestNeighbors_GreyCon_Kernel_RGBA << < NN_Grid, NN_Block, 0, stream[s] >> >
                    (d_big_img_nn[s], d_big_img_nn_grey[s], d_RGBA_img[s], big_width, big_height, width, height, scale);

                bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA << < BiCubic_Grid, BiCubic_Block, BiCubic_Shared_Mem_Size, stream[s] >> >
                    (d_big_img_bic[s], d_big_img_bic_grey[s], d_RGBA_img[s], big_width, big_height, width, height, scale);

                //PHASE 3 : Image Artifact Detection
                Artifact_Shared_Memory_Kernel << < Arti_Grid, Arti_Block, Arti_Shared_Mem_Size, stream[s] >> >
                    (d_big_artifact_map[s], d_big_img_nn_grey[s], d_big_img_bic_grey[s], big_width, big_height);

                //PHASE 4 : Artifact Map Post Processing
                horizontalGuassianBlurConvolve << < h_Guas_Grid, h_Guas_Block, h_Gauss_Mem, stream[s] >> >
                    (d_big_blurred_artifact_map_inter[s], d_big_artifact_map[s], big_width, big_height, GUAS_Ksize);

                verticalGuassianBlurConvolve << < v_Guas_Grid, v_Guas_Block, v_Gauss_Mem, stream[s] >> >
                    (d_big_blurred_artifact_map[s], d_big_blurred_artifact_map_inter[s], big_width, big_height, 0.05, GUAS_Ksize);

                //PHASE 5 : Image Fusion
                Image_Fusion_Kernel_RGBA << < RGB_Grid, RGB_Block, 0, stream[s] >> >
                    (d_big_rgba_img_fused[s], d_big_img_nn[s], d_big_img_bic[s], d_big_blurred_artifact_map[s], big_width, big_height);

                //PHASE 6 : Image Post Processing -> Convert Into Original Data Type
                rgbaToRGB_Kernel << < RGB_Grid, RGB_Block, rgbaToRGB_Shared_Mem_Size, stream[s] >> >
                    (d_big_img_fused[s], d_big_rgba_img_fused[s], big_width * big_height);

                //Send Device Images to Host
                cudaMemcpyAsync(h_big_img_fused[s], d_big_img_fused[s], sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost, stream[s]);

            }

            current_img += STREAM_COUNT;

            auto end = std::chrono::high_resolution_clock::now();
            auto dur = end - start;

            processing_time += std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();

            if (firstImg)
            {
                window = SDL_CreateWindow("PPM Image", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, big_width, big_height, SDL_WINDOW_SHOWN);
                if (!window) {
                    printf("Window creation failed: %c\n", SDL_GetError());
                    RUNNING = false;
                }

                renderer = SDL_CreateRenderer(window, -1, 0);
                if (!renderer) {
                    printf("Renderer creation failed: %c \n", SDL_GetError());
                    RUNNING = false;
                }
                firstImg = false;
            }

            texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STATIC, big_width, big_height);
            if (!texture)
            {
                printf("Texture creation failed: %c \n", SDL_GetError());
                RUNNING = false;
            }

            SDL_UpdateTexture(texture, nullptr, h_big_img_fused[0], big_width * 3);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);

            fps_msg = TTF_RenderText_Solid(Sans, fps_str, White);
            fps_txt = SDL_CreateTextureFromSurface(renderer, fps_msg);

            SDL_RenderCopy(renderer, fps_txt, NULL, &Message_rect);

            SDL_RenderPresent(renderer);

            SDL_PollEvent(&event);
            SDL_DestroyTexture(texture);

            SDL_FreeSurface(fps_msg);
            SDL_DestroyTexture(fps_txt);

            count++;
            current_img++;

            if (current_img > max_image)
                current_img = 1;
        }

        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();

    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    cudaDeviceReset();
    return 0;
}