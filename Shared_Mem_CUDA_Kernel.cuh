#define _USE_MATH_DEFINES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>

#include <fstream>
#include <string>

#include "naive_cuda.cuh"
#include "basic_optimization_cuda.cuh"
#include "shared_memory_cuda.cuh"
#include "ppm_image.cuh"
#include "util.cuh"

#include <chrono>

#ifdef USE_SDL

    #include <SDL.h>
    #undef main
    #include <SDL_ttf.h>
    #undef main

int sharedMemCudaOptimizedExecution()
{
    int width;
    int height;

    int big_width;
    int big_height;

    float diff;

    //Host Array Pointers
    unsigned char* h_img;
    RGBA_t* h_big_img_nn;
    RGBA_t* h_big_img_bic;
    RGBA_t* h_big_img_fused;

    //Device Array Pointers
    unsigned char* d_img;
    RGBA_t* d_RGBA_img;
    RGBA_t* d_big_img_nn;
    RGBA_t* d_big_img_bic;
    unsigned char* d_big_img_nn_grey;
    unsigned char* d_big_img_bic_grey;

    float* big_artifact_map_cuda;
    float* big_artifact_blurred_map_cuda;

    RGBA_t* big_rgba_img_fused_cuda;
    unsigned char* big_img_fused_cuda;

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

        sprintf(file_name, "./LM_Frame/image%d.ppm", current_img);
        h_img = (unsigned char*)readPPM(file_name, &width, &height);
        free(h_img);

        big_width = width * scale; big_height = height * scale;
        h_big_img_nn = (RGBA_t*)malloc(sizeof(RGBA_t) * big_width * big_height);
        h_big_img_bic = (RGBA_t*)malloc(sizeof(RGBA_t) * big_width * big_height);
        h_big_img_fused = (RGBA_t*)malloc(sizeof(RGBA_t) * big_width * big_height);

        int big_pixel_count = big_width * big_height;


        //Original Image & RGBA Image
        if (cudaMalloc((void**)&d_img, width * height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_RGBA_img, width * height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "RGBA Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Upscaled Images
        if (cudaMalloc((void**)&d_big_img_nn, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_big_img_bic, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Grey Versions for Upscaled Images
        if (cudaMalloc((void**)&d_big_img_nn_grey, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "NN Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_big_img_bic_grey, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "BIC Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Maps for Fusion
        if (cudaMalloc((void**)&big_artifact_map_cuda, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&big_artifact_blurred_map_cuda, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "Blured Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Final Image and RGBA Image
        if (cudaMalloc((void**)&big_img_fused_cuda, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&big_rgba_img_fused_cuda, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));



        while (RUNNING && event.type != SDL_QUIT)
        {
            if (count == frame_cap)
            {
                diff = 1000 * frame_cap / processing_time;
                sprintf(fps_str, "FPS:%.*f", 3, diff);

                count = 0;
                processing_time = 0;
            }

            sprintf(file_name, "./LM_Frame/image%d.ppm", current_img);

            h_img = (unsigned char*)readPPM(file_name,  &width, &height);

            auto start = std::chrono::high_resolution_clock::now();
            cudaDeviceSynchronize();

            cudaMemcpy(d_img, h_img, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            dim3 Grid(((big_width - 1) / block_dim) + 1, ((big_height - 1) / block_dim) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double

            dim3 Grid2(((width - 1) / block_dim) + 1, ((height - 1) / block_dim) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
            dim3 Block(block_dim, block_dim);

            dim3 Grid_Arti(((width - 1) / 8) + 1, ((height - 1) / 8) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
            dim3 Block_Arti(8, 8);

            dim3 BiCubic_Grid(((big_width - 1) / 8) + 1, ((big_height - 1) / 8) + 1);
            dim3 BiCubic_Block(8, 8);

            //Convert original image to RGBA image
            rgbToRGBA_Kernel <<< ceil((width * height)/256.0), 256 >>> (d_RGBA_img, d_img, width * height);

            //Launch the kernel and pass device matricies and size information
            bicubicInterpolation_GreyCon_Kernel_RGBA <<< Grid, Block >>> (d_big_img_bic, d_big_img_bic_grey, d_RGBA_img, big_width, big_height, width, height, scale);
            nearestNeighbors_shared_memory_Kernel << < Grid2, Block,  sizeof(RGBA_t) * block_dim * block_dim >> > (d_big_img_nn, d_big_img_nn_grey, d_RGBA_img, big_width, big_height, width, height, scale);
            //bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA <<<BiCubic_Grid, BiCubic_Block>>> (d_big_img_bic, d_big_img_bic_grey, d_RGBA_img, big_width, big_height, width, height, scale);
            
            //nearestNeighbors_shared_memory_one_thread_per_pixel_Kernel << < Grid, Block, block_dim * sizeof(unsigned char) >> >(big_img_nn_cuda, big_img_nn_grey_cuda, img_cuda, big_width, big_height, const_width, const_height, scale);
            Artifact_Shared_Memory_Kernel << < Grid_Arti, Block_Arti, sizeof(float) * 8 * 8 >> > (big_artifact_map_cuda, d_big_img_nn_grey, d_big_img_bic_grey, big_width, big_height);

            //Artifact_Grey_Kernel <<< Grid, Block >>> (big_artifact_map_cuda, d_big_img_nn_grey, d_big_img_bic_grey, big_width, big_height);
            //GuassianBlur_Threshold_Map_Kernel <<< Grid, Block >>> (big_artifact_blurred_map_cuda, big_artifact_map_cuda, big_width, big_height, 3, 1.5, 0.05);
            //Image_Fusion_Kernel_RGBA <<< Grid, Block >>> (big_rgba_img_fused_cuda, d_big_img_nn, d_big_img_bic, big_artifact_blurred_map_cuda, big_width, big_height);
 
            rgbaToRGB_Kernel << < ceil((big_width * big_height) / 256.0), 256 >> > (big_img_fused_cuda, d_big_img_nn, big_width * big_height);

            //rgbaToRGB_Kernel <<< ceil((big_width * big_height) / 256.0), 256 >>> (big_img_fused_cuda, big_rgba_img_fused_cuda, big_width * big_height);
            cudaDeviceSynchronize();

            cudaMemcpy(h_big_img_fused, big_img_fused_cuda, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);

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

            SDL_UpdateTexture(texture, nullptr, h_big_img_fused, big_width * 3);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);

            fps_msg = TTF_RenderText_Solid(Sans, fps_str, White);
            fps_txt = SDL_CreateTextureFromSurface(renderer, fps_msg);

            SDL_RenderCopy(renderer, fps_txt, NULL, &Message_rect);

            SDL_RenderPresent(renderer);

            SDL_PollEvent(&event);
            SDL_DestroyTexture(texture);

            SDL_FreeSurface(fps_msg);
            SDL_DestroyTexture(fps_txt);

            free(h_img); 
            
            count++;
            current_img++;

            if (current_img > max_image)
                current_img = 1;
        }

        free(h_big_img_nn);               free(h_big_img_bic);              free(h_big_img_fused);

        cudaFree(d_img);                         cudaFree(d_big_img_nn);          cudaFree(d_big_img_bic);
        cudaFree(d_big_img_nn_grey);             cudaFree(d_big_img_bic_grey);    cudaFree(big_artifact_map_cuda);
        cudaFree(big_artifact_blurred_map_cuda);    cudaFree(big_img_fused_cuda);


        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();

        //free(width);    free(height);
    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    cudaDeviceReset();
    return 0;
}

#else
int sharedMemCudaOptimizedExecution()
{
    int width;
    int height;

    int big_width;
    int big_height;
    int big_pixel_count;

    //Host Array Pointers, these should always be unsigned char
    unsigned char*      h_img;                              //Original Small Input Image
    unsigned char*  h_big_img_nn;                       //Upscaled Nearest Neighbor Image
    unsigned char*  h_big_img_nn_grey;                  //Upscaled Greyscale Nearest Neighbor Image
    unsigned char*  h_big_img_bic;                      //Upscaled Bicubic Image
    unsigned char*  h_big_img_bic_grey;                 //Upscaled Greyscale Bicubic Image
    unsigned char*  h_big_img_DIFF_grey;                //Upscaled Greyscale Difference Image
    unsigned char*  h_big_img_SSIM_grey;                //Upscaled Greyscale SSIM Image
    unsigned char*  h_big_img_ARTIFACT_grey;            //Upscaled Greyscale ARTIFACT Image
    unsigned char*  h_big_img_BLURRED_ARTIFACT_grey;    //Upscaled Greyscale BLURRED ARTIFACT Image
    unsigned char*  h_big_img_fused;                    //Upscaled Fused Image
    float*          h_diff_map;                         //Difference Map
    float*          h_ssim_map;                         //SSIM Map
    float*          h_artifact_map;                     //Artifact Map
    float*          h_blurred_artifact_map;             //Blurred Artifact Map
    //Temporary Images for debug
    unsigned char*      h_temp_output_img1;
    unsigned char*      h_temp_output_img2;

    //Device Array Pointers
    unsigned char*      d_img;                              //Original Small Input Image
    RGBA_t*             d_RGBA_img;                         //Original Small Input Image w/ 32bit-pixel format
    RGBA_t*             d_big_img_nn;                       //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    RGBA_t*             d_big_img_bic;                      //Upscaled Bicubic Image w/ 32bit-pixel format
    unsigned char*      d_big_img_nn_grey;                  //Upscaled Greyscale Nearest Neighbor Image
    unsigned char*      d_big_img_bic_grey;                 //Upscaled Greyscale Bicubic Image
    float*              d_big_artifact_map;                 //Upscaled Artifact Map for image fusion
    float*              d_big_blurred_artifact_map;         //Upscaled Blurred Artifact Map for image fusion
    RGBA_t*             d_big_rgba_img_fused;               //Upscaled Fused Image w/ 32bit-pixel format
    unsigned char*      d_big_img_fused;                    //Upscaled Fused Image  
    //Temporary Images for debug
    unsigned char*      d_temp_output_img1;
    unsigned char*      d_temp_output_img2;

    //Kernel Parameters
    int scale = 3;
    bool RUNNING = true;
    bool firstImg = true;

    //Not sure these are needed will keep for now
    int block_dim = 8; //The x and y axis size for the block is 16 threads. Total 256 threads
    int window_size = 8;


    //Lets start off with timing one image
    try
    {
        //Check that CUDA-capable GPU is installed
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        }

        //***** Temp *****//
        char fps_str[50];
        char file_name[50];
        int count = 0;

        double frame_cap = 10;
        sprintf(fps_str, "FPS:%.*f", 3, 0.0);

        int max_image = 200;
        int current_img = 37;

        double processing_time = 0;
        //***** Temp *****//


        //Read in first image initially to get input width and height.
        //sprintf(file_name, "./LM_Frame/image%d.ppm", current_img);
        sprintf(file_name, "./LAD/LAD_%d.ppm", current_img);
        h_img = (unsigned char*)readPPM(file_name, &width, &height);
        free(h_img);

        //Define big image width and height
        big_width       = width * scale; big_height = height * scale;
        big_pixel_count = big_width * big_height;
        
        //******** Malloc Host Images ********//
        h_big_img_nn                    = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        h_big_img_nn_grey               = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_bic                   = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        h_big_img_bic_grey              = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_ARTIFACT_grey         = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_BLURRED_ARTIFACT_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_fused                 = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        h_artifact_map                  = (float*)malloc(sizeof(float) * big_pixel_count);
        h_blurred_artifact_map          = (float*)malloc(sizeof(float) * big_pixel_count);
        h_temp_output_img1   = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        h_temp_output_img2   = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        //******** Malloc Host Images ********//

        //******** Malloc Device Images ********//

        //Original Image & RGBA Image
        if (cudaMalloc((void**)&d_img, width * height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_RGBA_img, width * height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "RGBA Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Upscaled Images
        if (cudaMalloc((void**)&d_big_img_nn, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_big_img_bic, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Grey Versions for Upscaled Images
        if (cudaMalloc((void**)&d_big_img_nn_grey, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "NN Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_big_img_bic_grey, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "BIC Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Maps for Fusion
        if (cudaMalloc((void**)&d_big_artifact_map, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_big_blurred_artifact_map, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "Blured Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Final Image and RGBA Image
        if (cudaMalloc((void**)&d_big_img_fused, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_big_rgba_img_fused, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Temporary Output Images for comparison and debug
        if(cudaMalloc((void**)&d_temp_output_img1, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if(cudaMalloc((void**)&d_temp_output_img2, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));


        //**************** Setup Kernel ****************//
        //sprintf(file_name, "./LM_Frame/image%d.ppm", current_img);

        dim3 Grid(((big_width - 1) / block_dim) + 1, ((big_height - 1) / block_dim) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double

        dim3 Grid2(((width - 1) / block_dim) + 1, ((height - 1) / block_dim) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 Block(block_dim, block_dim);

        dim3 Grid_Arti(((width - 1) / 8) + 1, ((height - 1) / 8) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 Block_Arti(8, 8);

        dim3 BiCubic_Block(12, 12);
        dim3 BiCubic_Grid(((big_width - 1) / BiCubic_Block.x) + 1, ((big_height - 1) / BiCubic_Block.y) + 1);
        int BiCubic_Shared_Mem_Size = ((BiCubic_Block.y / scale) + 3) * ((BiCubic_Block.x / scale) + 3);


        dim3 GRID_RGB_Convert(ceil((big_width * big_height) / 256.0));
        dim3 BLOCK_RGB_Convert(256);
        //**************** Setup Kernel ****************//

        //Variables for timing
        cudaEvent_t astartEvent, astopEvent;
        float aelapsedTime;
        cudaEventCreate(&astartEvent);
        cudaEventCreate(&astopEvent);
        
        //**************** New Bicubic Stuff ****************//
       
        //Generate Bicubic Kernel
        int BIC_Ksize = 4*scale;
        float *h_bic_kernel = (float*)malloc(sizeof(float) * BIC_Ksize);
        float *d_bic_kernel;
        if (cudaMalloc((void**)&d_bic_kernel, BIC_Ksize * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "BIC Kernel Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));


        float sum = 0;
        for(int i = 0; i < BIC_Ksize; i++)
        {
            float x = -2.0 + 4.0 * i / (BIC_Ksize-1);
            h_bic_kernel[i] = cubicKernel(x);
            sum += h_bic_kernel[i];
        }
        //Normalize Kernel
        for(int i = 0; i < BIC_Ksize; i++)
        {
            h_bic_kernel[i] /= sum;
        }
        cudaMemcpy(d_bic_kernel, h_bic_kernel, BIC_Ksize *sizeof(float), cudaMemcpyHostToDevice);
        //**************** New Bicubic Stuff ****************//

        //**************** New Guassian Blur ****************//
        int GUAS_Ksize = 7;
        float GUAS_Sigma = 1.5;

        int kernel_center = GUAS_Ksize / 2;
        
        //sum = 0;

        ////Define Host & Device  Side Guassian Kernel
        //float *h_guas_kernel = (float*)malloc(sizeof(float) * GUAS_Ksize * GUAS_Ksize);
        ////float *d_guas_kernel;
        //if (cudaMalloc((void**)&d_guas_kernel, GUAS_Ksize * sizeof(float)) != cudaSuccess)
        //    fprintf(stderr, "GUAS Kernel Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //for (int x = 0; x < GUAS_Ksize; x++)
        //{
        //    double exponent = -((x - kernel_center) * (x - kernel_center)) / (2 * GUAS_Sigma * GUAS_Sigma);
        //    h_guas_kernel[x] = exp(exponent) / sqrt((2 * M_PI * GUAS_Sigma));
        //    sum += h_guas_kernel[x];
        //}
        ////Normalize
        //for (int i = 0; i < GUAS_Ksize; i++)
        //        h_guas_kernel[i] /= sum;

        ////Define Host & Device  Side Guassian Kernel
        //float *h_guas_kernel = (float*)malloc(sizeof(float) * GUAS_Ksize * GUAS_Ksize);
        ////float *d_guas_kernel;
        //if (cudaMalloc((void**)&d_guas_kernel, GUAS_Ksize * GUAS_Ksize * sizeof(float)) != cudaSuccess)
        //    fprintf(stderr, "GUAS Kernel Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //for (int y = 0; y < GUAS_Ksize; y++)
        //{
        //    for (int x = 0; x < GUAS_Ksize; x++)
        //    {
        //        double exponent = -((x - kernel_center) * (x - kernel_center) - (y - kernel_center) * (y - kernel_center)) / (2 * GUAS_Sigma * GUAS_Sigma);
        //        h_guas_kernel[y * GUAS_Ksize + x] = exp(exponent) / (2 * M_PI * GUAS_Sigma * GUAS_Sigma);
        //        sum += h_guas_kernel[y * GUAS_Ksize + x];
        //    }
        //}
        ////Normalize
        //for (int i = 0; i < GUAS_Ksize; i++)
        //    for (int j = 0; j < GUAS_Ksize; j++)
        //        h_guas_kernel[i * GUAS_Ksize + j] /= sum;


        //cudaMemcpyToSymbol(d_guas_kernel, h_guas_kernel, GUAS_Ksize * GUAS_Ksize *sizeof(float));
        //*************** New Guassian Blur *****************//



        //**************** Run & Time Kernels ****************//
        //cudaEventRecord(astartEvent, 0);

        //Load Input Image
        h_img = (unsigned char*)readPPM(file_name, &width, &height);
            
        //Copy Input Image to Device
        cudaMemcpy(d_img, h_img, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        //Convert original image to RGBA image
        rgbToRGBA_Kernel <<< GRID_RGB_Convert, BLOCK_RGB_Convert >> > (d_RGBA_img, d_img, width * height);

        //Upscale image and convert to greyscale using Bicubic method
        //horizontalBicubicConvolve<<<Grid, Block>>>(d_big_img_bic, d_RGBA_img, d_bic_kernel, big_width, big_height, width, height, scale, BIC_Ksize );
        //verticalBicubicConvolve<<<Grid, Block>>>(d_big_img_bic, d_big_img_bic_grey, d_RGBA_img, d_bic_kernel, big_width, big_height, width, height, scale, BIC_Ksize );
        bicubicInterpolation_GreyCon_Kernel_RGBA <<< Grid, Block >>> (d_big_img_bic, d_big_img_bic_grey, d_RGBA_img, big_width, big_height, width, height, scale);
        //bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA<<<BiCubic_Grid, BiCubic_Block, sizeof(RGBA_t) * BiCubic_Shared_Mem_Size>>> (d_big_img_bic, d_big_img_bic_grey, d_RGBA_img, big_width, big_height, width, height, scale);
        cudaDeviceSynchronize();

        //Upscale image and convert to greyscale using Nearest Neighbor method
        //nearestNeighbors_shared_memory_one_thread_per_pixel_Kernel <<< Grid, Block, sizeof(RGBA_t) * block_dim * block_dim / scale >>> (d_big_img_nn, d_big_img_nn_grey, d_RGBA_img, big_width, big_height, width, height, scale);
        nearestNeighbors_GreyCon_Kernel_RGBA <<< Grid, Block >>> (d_big_img_nn, d_big_img_nn_grey, d_RGBA_img, big_width, big_height, width, height, scale);
        
        //Artifact_Shared_Memory_Kernel << < Grid_Arti, Block_Arti, sizeof(float) * 8 * 8 >> > (d_big_artifact_map, d_big_img_nn_grey, d_big_img_bic_grey, big_width, big_height);
        Artifact_Grey_Kernel <<< Grid, Block >>>                (d_big_artifact_map         , d_big_img_nn_grey             , d_big_img_bic_grey        , big_width, big_height);
        
        dim3 h_Guas_Block(32, 32);
        dim3 h_Guas_Grid(((big_width - 1) / h_Guas_Block.x) + 1, ((big_height - 1) / h_Guas_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        
        dim3 v_Guas_Block(32, 32);
        dim3 v_Guas_Grid(((big_width - 1) / v_Guas_Block.x) + 1, ((big_height - 1) / v_Guas_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double


        cudaEventRecord(astartEvent, 0);
        //GuassianBlur_Threshold_Map_Kernel <<< Grid, Block >>>   (d_big_blurred_artifact_map , d_big_artifact_map                                        , big_width, big_height, 3, 1.5, 0.05);
        //GuassianBlur_Threshold_Map_Shared_Memory_Kernel<<< Grid, Block >>>(d_big_blurred_artifact_map, d_big_artifact_map, d_guas_kernel, big_width, big_height, 0.05, GUAS_Ksize);
        //GuassianBlur_Threshold_Map_Constant_Memory_Kernel<<< Grid, Block >>>(d_big_blurred_artifact_map, d_big_artifact_map, big_width, big_height, 0.05, GUAS_Ksize);
        
        horizontalGuassianBlurConvolve  <<< h_Guas_Grid, h_Guas_Block, sizeof(float) * (h_Guas_Block.x + GUAS_Ksize - 1) * h_Guas_Block.y >>>(d_big_blurred_artifact_map, d_big_artifact_map, big_width, big_height, GUAS_Ksize);
        //cudaEventRecord(astartEvent, 0);
        //cudaEventRecord(astopEvent, 0);
        verticalGuassianBlurConvolve    <<< v_Guas_Grid, v_Guas_Block, sizeof(float) * (v_Guas_Block.y + GUAS_Ksize - 1) * v_Guas_Block.x >>>(d_big_blurred_artifact_map, d_big_blurred_artifact_map, big_width, big_height, 0.05, GUAS_Ksize);
        //GuassianBlurConvolve<<< Grid, Block >>>(d_big_blurred_artifact_map, d_big_artifact_map, big_width, big_height, 0.05, GUAS_Ksize);
        cudaEventRecord(astopEvent, 0);
        
        //Fusion
        Image_Fusion_Kernel_RGBA <<< Grid, Block >>>            (d_big_rgba_img_fused       , d_big_img_nn, d_big_img_bic   , d_big_blurred_artifact_map, big_width, big_height);

        //Convert Upscaled image back to RGB
        rgbaToRGB_Kernel <<< GRID_RGB_Convert, BLOCK_RGB_Convert >>> (d_big_img_fused, d_big_rgba_img_fused, big_width * big_height);

        //Send Device Images to Host
        cudaMemcpy(h_big_img_fused, d_big_img_fused, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);

        //cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("Total compute time (ms) %f\n", aelapsedTime);
        //**************** Run & Time Kernels ****************//


        //Convert Intermidiate Images to RGB and send them to the host
        rgbaToRGB_Kernel <<< GRID_RGB_Convert, BLOCK_RGB_Convert >>> (d_temp_output_img1, d_big_img_nn, big_width * big_height);
        cudaMemcpy(h_big_img_nn, d_temp_output_img1, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);

        rgbaToRGB_Kernel <<< GRID_RGB_Convert, BLOCK_RGB_Convert >>> (d_temp_output_img1, d_big_img_bic, big_width * big_height);
        cudaMemcpy(h_big_img_bic, d_temp_output_img1, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);

        //Send Device Images to Host
        
        cudaMemcpy(h_big_img_nn_grey        , d_big_img_nn_grey         , sizeof(unsigned char) * big_width * big_height    , cudaMemcpyDeviceToHost);
        cudaMemcpy(h_big_img_bic_grey       , d_big_img_bic_grey        , sizeof(unsigned char) * big_width * big_height    , cudaMemcpyDeviceToHost);
        cudaMemcpy(h_artifact_map           , d_big_artifact_map        , sizeof(float) * big_width * big_height    , cudaMemcpyDeviceToHost);
        cudaMemcpy(h_blurred_artifact_map   , d_big_blurred_artifact_map, sizeof(float) * big_width * big_height    , cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        //Convert Maps to Greyscale
        Map2Greyscale(h_big_img_ARTIFACT_grey           , h_artifact_map        , big_width, big_height, 255);   //Artifact values should be between 0-255;
        Map2Greyscale(h_big_img_BLURRED_ARTIFACT_grey   , h_blurred_artifact_map, big_width, big_height, 255);   //Artifact values should be between 0-255;
      
        //Save Images
        writePPM    ("./Shared_Memory_Output/NN.ppm"                   , (char*)h_big_img_nn                       , big_width, big_height);
        writePPMGrey("./Shared_Memory_Output/NN_Grey.ppm"              , (char*)h_big_img_nn_grey                  , big_width, big_height);
        writePPM    ("./Shared_Memory_Output/BIC.ppm"                  , (char*)h_big_img_bic                      , big_width, big_height);
        writePPMGrey("./Shared_Memory_Output/BIC_Grey.ppm"             , (char*)h_big_img_bic_grey                 , big_width, big_height);
        writePPMGrey("./Shared_Memory_Output/ARTIFACT_Grey.ppm"        , (char*)h_big_img_ARTIFACT_grey            , big_width, big_height);
        writePPMGrey("./Shared_Memory_Output/BLURRED_ARTIFACT_Grey.ppm", (char*)h_big_img_BLURRED_ARTIFACT_grey    , big_width, big_height);
        writePPM    ("./Shared_Memory_Output/FUSED.ppm"                , (char*)h_big_img_fused                    , big_width, big_height);


        //Compare with Serial Image
        h_temp_output_img1 = (unsigned char*)readPPM("./Serial_Output/NN.ppm", &big_width, &big_height);
        Image_Compare(h_temp_output_img1, h_big_img_nn, big_width, big_height);
        
        h_temp_output_img1 = (unsigned char*)readPPM("./Serial_Output/BIC.ppm", &big_width, &big_height);
        Image_Compare(h_temp_output_img1, h_big_img_bic, big_width, big_height);

        h_temp_output_img1 = (unsigned char*)readPPM("./Serial_Output/ARTIFACT_Grey.ppm", &big_width, &big_height);
        Grey_Image_Compare(h_temp_output_img1, h_big_img_ARTIFACT_grey, big_width, big_height);

        h_temp_output_img1 = (unsigned char*)readPPM("./Serial_Output/BLURRED_ARTIFACT_Grey.ppm", &big_width, &big_height);
        Grey_Image_Compare(h_temp_output_img1, h_big_img_BLURRED_ARTIFACT_grey, big_width, big_height);

        h_temp_output_img1 = (unsigned char*)readPPM("./Serial_Output/FUSED.ppm", &big_width, &big_height);
        Image_Compare(h_temp_output_img1, h_big_img_fused, big_width, big_height);
        
        free(h_temp_output_img1);

        //Free Host Memory
        free(h_img);
        free(h_big_img_nn);
        free(h_big_img_bic);   
        free(h_big_img_fused);

        //Free device Memory
        cudaFree(d_img);
        cudaFree(d_RGBA_img);
        cudaFree(d_big_img_nn);
        cudaFree(d_big_img_bic);
        cudaFree(d_big_img_nn_grey);
        cudaFree(d_big_img_bic_grey);
        cudaFree(d_big_artifact_map);
        cudaFree(d_big_blurred_artifact_map);
        cudaFree(d_big_rgba_img_fused);
        cudaFree(d_big_img_fused);
        cudaFree(d_temp_output_img1);
        cudaFree(d_temp_output_img2);

        

#if 0
        while (RUNNING)
        {
            if (count == max_image)
            {
                diff = 1000 * count / processing_time;
                printf("FPS:%.*f\n", 3, diff);

                count = 0;
                processing_time = 0;
                RUNNING = false;
            }

            sprintf(file_name, "./LM_Frame/image%d.ppm", current_img);

            h_img = (unsigned char*)readPPM(file_name, &width, &height);

            auto start = std::chrono::high_resolution_clock::now();
            cudaDeviceSynchronize();

            cudaMemcpy(d_img, h_img, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();


            //Convert original image to RGBA image
            rgbToRGBA_Kernel << < ceil((width * height) / 256.0), 256 >> > (d_RGBA_img, d_img, width * height);

            //Launch the kernel and pass device matricies and size information
            bicubicInterpolation_GreyCon_Kernel_RGBA << < Grid, Block >> > (d_big_img_bic, d_big_img_bic_grey, d_RGBA_img, big_width, big_height, width, height, scale);
            nearestNeighbors_shared_memory_Kernel << < Grid2, Block >> > (d_big_img_nn, d_big_img_nn_grey, d_RGBA_img, big_width, big_height, width, height, scale);
            //bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA <<<BiCubic_Grid, BiCubic_Block>>> (d_big_img_bic, d_big_img_bic_grey, d_RGBA_img, big_width, big_height, width, height, scale);

            //nearestNeighbors_shared_memory_one_thread_per_pixel_Kernel << < Grid, Block, block_dim * sizeof(unsigned char) >> >(big_img_nn_cuda, big_img_nn_grey_cuda, img_cuda, big_width, big_height, const_width, const_height, scale);
            Artifact_Shared_Memory_Kernel << < Grid_Arti, Block_Arti, sizeof(float) * 8 * 8 >> > (big_artifact_map_cuda, d_big_img_nn_grey, d_big_img_bic_grey, big_width, big_height);

            //Artifact_Grey_Kernel <<< Grid, Block >>> (big_artifact_map_cuda, d_big_img_nn_grey, d_big_img_bic_grey, big_width, big_height);
            GuassianBlur_Threshold_Map_Kernel << < Grid, Block >> > (big_artifact_blurred_map_cuda, big_artifact_map_cuda, big_width, big_height, 3, 1.5, 0.05);
            Image_Fusion_Kernel_RGBA << < Grid, Block >> > (big_rgba_img_fused_cuda, d_big_img_nn, d_big_img_bic, big_artifact_blurred_map_cuda, big_width, big_height);

            rgbaToRGB_Kernel << < ceil((big_width * big_height) / 256.0), 256 >> > (big_img_fused_cuda, big_rgba_img_fused_cuda, big_width * big_height);
            cudaDeviceSynchronize();

            cudaMemcpy(h_big_img_fused, big_img_fused_cuda, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);

            auto end = std::chrono::high_resolution_clock::now();
            auto dur = end - start;

            processing_time += std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();

            free(h_img);

            count++;
            current_img++;

            if (current_img > max_image)
                current_img = 1;
        }

        free(h_big_img_nn);               free(h_big_img_bic);              free(h_big_img_fused);

        cudaFree(d_img);                         cudaFree(d_big_img_nn);          cudaFree(d_big_img_bic);
        cudaFree(d_big_img_nn_grey);             cudaFree(d_big_img_bic_grey);    cudaFree(big_artifact_map_cuda);
        cudaFree(big_artifact_blurred_map_cuda);    cudaFree(big_img_fused_cuda);

        //free(width);    free(height);
#endif

    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    cudaDeviceReset();
    return 0;
}
#endif 