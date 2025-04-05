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

#include <chrono>

#include <SDL.h>
#undef main
#include <SDL_ttf.h>
#undef main

int sharedMemCudaOptimizedExecution()
{
    int* width;
    int* height;

    int const_width;
    int const_height;

    int big_width;
    int big_height;

    float diff;

    //Host Array Pointers
    unsigned char* img;
    unsigned char* big_img_nn;
    unsigned char* big_img_bic;
    unsigned char* big_img_fused;

    //Device Array Pointers
    unsigned char* img_cuda;
    unsigned char* big_img_nn_cuda;
    unsigned char* big_img_bic_cuda;
    unsigned char* big_img_nn_grey_cuda;
    unsigned char* big_img_bic_grey_cuda;

    float* big_artifact_map_cuda;
    float* big_artifact_blurred_map_cuda;

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
        width = (int*)malloc(sizeof(int));
        height = (int*)malloc(sizeof(int));

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

            img = (unsigned char*)readPPM(file_name, width, height);

            auto start = std::chrono::high_resolution_clock::now();

            const_width = *width;
            const_height = *height;

            big_width = const_width * scale; big_height = const_height * scale;
            big_img_nn = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height * 3);
            big_img_bic = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height * 3);
            big_img_fused = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height * 3);

            int big_pixel_count = big_width * big_height;

            cudaDeviceSynchronize();

            //Original Image
            if (cudaMalloc((void**)&img_cuda, const_width * const_height * sizeof(unsigned char) * 3) != cudaSuccess)
                printf("Original Image Failed To Copy To Device.\n");      //Notify failure

            //Upscaled Images
            if (cudaMalloc((void**)&big_img_nn_cuda, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
                fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&big_img_bic_cuda, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
                fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            //Grey Versions for Upscaled Images
            if (cudaMalloc((void**)&big_img_nn_grey_cuda, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
                fprintf(stderr, "NN Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&big_img_bic_grey_cuda, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
                fprintf(stderr, "BIC Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            //Maps for Fusion
            if (cudaMalloc((void**)&big_artifact_map_cuda, big_width * big_height * sizeof(float)) != cudaSuccess)
                fprintf(stderr, "Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&big_artifact_blurred_map_cuda, big_width * big_height * sizeof(float)) != cudaSuccess)
                fprintf(stderr, "Blured Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            //Final Image
            if (cudaMalloc((void**)&big_img_fused_cuda, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
                fprintf(stderr, "Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            cudaMemcpy(img_cuda, img, sizeof(unsigned char) * const_width * const_height * 3, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            dim3 Grid(((big_width - 1) / block_dim) + 1, ((big_height - 1) / block_dim) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
            dim3 Block(block_dim, block_dim);

            //Launch the kernel and pass device matricies and size information
            bicubicInterpolation_GreyCon_Kernel << < Grid, Block >> > (big_img_bic_cuda, big_img_nn_grey_cuda, img_cuda, big_width, big_height, const_width, const_height, scale);
            nearestNeighbors_GreyCon_Kernel << < Grid, Block >> > (big_img_nn_cuda, big_img_nn_grey_cuda, img_cuda, big_width, big_height, const_width, const_height, scale);
            //nearestNeighbors_shared_memory_one_thread_per_pixel_Kernel << < Grid, Block, block_dim * sizeof(unsigned char) >> >(big_img_nn_cuda, big_img_nn_grey_cuda, img_cuda, big_width, big_height, const_width, const_height, scale);
            Artifact_Grey_Kernel << < Grid, Block >> > (big_artifact_map_cuda, big_img_nn_grey_cuda, big_img_bic_grey_cuda, big_width, big_height);
            GuassianBlur_Threshold_Map_Kernel << < Grid, Block >> > (big_artifact_blurred_map_cuda, big_artifact_map_cuda, big_width, big_height, 3, 1.5, 0.05);
            Image_Fusion_Kernel << < Grid, Block >> > (big_img_fused_cuda, big_img_nn_cuda, big_img_bic_cuda, big_artifact_blurred_map_cuda, big_width, big_height);
            cudaDeviceSynchronize();

            cudaMemcpy(big_img_fused, big_img_fused_cuda, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);

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

            SDL_UpdateTexture(texture, nullptr, big_img_fused, big_width * 3);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);

            fps_msg = TTF_RenderText_Solid(Sans, fps_str, White);
            fps_txt = SDL_CreateTextureFromSurface(renderer, fps_msg);

            SDL_RenderCopy(renderer, fps_txt, NULL, &Message_rect);

            SDL_RenderPresent(renderer);

            SDL_PollEvent(&event);
            SDL_DestroyTexture(texture);

            SDL_FreeSurface(fps_msg);
            SDL_DestroyTexture(fps_txt);

            free(img);                          free(big_img_nn);               free(big_img_bic);              free(big_img_fused);

            cudaFree(img_cuda);                         cudaFree(big_img_nn_cuda);          cudaFree(big_img_bic_cuda);
            cudaFree(big_img_nn_grey_cuda);             cudaFree(big_img_bic_grey_cuda);    cudaFree(big_artifact_map_cuda);
            cudaFree(big_artifact_blurred_map_cuda);    cudaFree(big_img_fused_cuda);

            count++;
            current_img++;

            if (current_img > max_image)
                current_img = 1;
        }

        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();

        free(width);    free(height);
    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    cudaDeviceReset();
    return 0;
}
