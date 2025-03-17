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

#include "helper.cuh"
#include "serial_code.cuh"
#include "naive_cuda.cuh"
#include "ppm_image.cuh"

#include <chrono>

#include <SDL.h>
#undef main
#include <SDL_ttf.h>
#undef main


void Artifact_Detection(unsigned char* mask_img, unsigned char* img_1, unsigned char* img_2, int* width, int* height, int* window_size, float TH)
{
    int num_windows_x = ceil(*width / (float) *window_size);
    int num_windows_y = ceil(*height / (float) *window_size);

    int total_win_pix = *window_size * *window_size;

    float* img1_window = (float*)malloc(sizeof(float) * total_win_pix * 3);
    float* img2_window = (float*)malloc(sizeof(float) * total_win_pix * 3);

    float* difference_window = (float*)malloc(sizeof(float) * *width * *height * 3);
    float* ssim_window = (float*)malloc(sizeof(float) * 3);

    int img_x = 0; 
    int img_y = 0;

    float* metric_img = (float*)malloc(sizeof(float) * total_win_pix * 3);
    float metric_temp;
    //(Metric < TH) ? 0 : 1

    for (int win_y = 0; win_y < num_windows_y; win_y++)
    {
        for (int win_x = 0; win_x < num_windows_x; win_x++)
        {
            //DATA LOADING PHASE
            for (int y = 0; y < *window_size; y++)
            {
                for (int x = 0; x < *window_size; x++)
                {
                    img_x = x + win_x * *window_size;
                    img_y = y + win_y * *window_size;

                    if ((img_y < *height) && (img_x < *width))
                    {
                        for (int chn = 0; chn < CHN_NUM; chn++)
                        {
                            img1_window[(y * *window_size + x) * CHN_NUM + chn] = (float) img_1[(img_y * *width + img_x) * CHN_NUM + chn];
                            img2_window[(y * *window_size + x) * CHN_NUM + chn] = (float) img_2[(img_y * *width + img_x) * CHN_NUM + chn];
                        }
                    }
                    else
                    {
                        for (int chn = 0; chn < CHN_NUM; chn++)
                        {
                            img1_window[(y * *window_size + x) * CHN_NUM + chn] = -1.0;
                            img2_window[(y * *window_size + x) * CHN_NUM + chn] = -1.0;
                        }
                    }
                }
            }

            //Image Difference Phase
            ABS_Difference(difference_window, img1_window, img2_window, window_size, window_size);
            SSIM(ssim_window, img1_window, img2_window, window_size, window_size);

            //Metric Calculations
            for (int y = 0; y < *window_size; y++)
            {
                for (int x = 0; x < *window_size; x++)
                {
                    img_x = x + win_x * *window_size;
                    img_y = y + win_y * *window_size;

                    if ((img_y < *height) && (img_x < *width))
                    {
                        for (int chn = 0; chn < CHN_NUM; chn++)
                        {
                            metric_temp = difference_window[(y * *window_size + x) * CHN_NUM + chn] * ssim_window[chn];
                            mask_img[(img_y * *width + img_x) * CHN_NUM + chn] = (unsigned char) metric_temp;
                            //mask_img/*metric_img*/[(y * *window_size + x) * CHN_NUM + chn]
                        }
                    }
                }
            }
            //Gausian Blur Stage

        }

    }

    //Reminder to free all the stuff
    free(img1_window); free(img2_window);
    free(difference_window);  free(ssim_window);
    free(metric_img);
    
}

int serialExecution()
{
    try
    {
        int* width = (int*)malloc(sizeof(int));
        int* height = (int*)malloc(sizeof(int));
        unsigned char* img;

        int big_width;
        int big_height;
        int window_size = 8;

        int scale = 2;

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

        bool RUNNING = true;
        bool firstImg = true;
        SDL_Window* window;
        SDL_Renderer* renderer;
        SDL_Texture* texture;
        SDL_Event event;
        SDL_PollEvent(&event);


        int const_width;
        int const_height;

        float diff = 0;
        unsigned char* big_img_nn;
        unsigned char* big_img_nn_grey;
        unsigned char* big_img_bic;
        unsigned char* big_img_bic_grey;
        unsigned char* big_img_dif;
        unsigned char* big_img_dif_grey;
        unsigned char* big_img_ssim_grey;


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
        int current_img = 60;

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
            big_img_nn_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height);
            big_img_bic = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height * 3);
            big_img_bic_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height);
            big_img_dif = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height * 3);
            big_img_dif_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height);
            big_img_ssim_grey = (unsigned char*)malloc(sizeof(unsigned char) * (big_width) * (big_height));
            //unsigned char* big_img_ssim = (unsigned char*)malloc(sizeof(unsigned char) * *big_width * *big_height * 3);

            //printf("Image dimensions: %d x %d\n", *width, *height);
            //printf("Upscale Image dimensions: %d x %d\n", *big_width, *big_height);

            //nearestNeighbors(big_img_nn, big_width, big_height, img, width, height, scale);
            nearestNeighbors(big_img_nn, big_width, big_height, img, const_width, const_height, scale);
            RGB2Greyscale(big_img_nn_grey, big_img_nn, big_width, big_height);
            bicubicInterpolation(big_img_bic, big_width, big_height, img, const_width, const_height, scale);
            RGB2Greyscale(big_img_bic_grey, big_img_bic, big_width, big_height);

            //ABS_Difference_Grey(big_img_dif_grey, big_img_nn_grey, big_img_bic_grey, big_width, big_height);
            //ABS_Difference(big_img_dif, big_img_nn, big_img_bic, big_width, big_height);
            //Artifact_Detection(big_img_dif, big_img_nn, big_img_bic, big_width, big_height, window_size, 0.9);
            //SSIM_Grey(big_img_ssim_grey, big_img_nn_grey, big_img_bic_grey, big_width, big_height);

            //WeightMap_Grey(big_img_wm_grey, big_img_dif_grey, big_img_ssim_grey, big_width, big_height);
            //writePPM("output_NN.ppm", (char*)big_img_nn, big_width, big_height);
            //writePPM("output_BIC.ppm", (char*)big_img_bic, big_width, big_height);
            //writePPM("output_diff.ppm", (char*)big_img_dif, big_width, big_height);

            auto end = std::chrono::high_resolution_clock::now();
            auto dur = end - start;

            processing_time += std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();

            if (firstImg)
            {

                writePPMGrey("output_NN_grey.ppm", (char*)big_img_nn_grey, big_width, big_height);
                writePPMGrey("output_BIC_grey.ppm", (char*)big_img_bic_grey, big_width, big_height);
                writePPMGrey("output_DIFF_grey.ppm", (char*)big_img_dif_grey, big_width, big_height);
                writePPMGrey("output_SSIM_grey.ppm", (char*)big_img_ssim_grey, big_width, big_height);

                window = SDL_CreateWindow("PPM Image", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, big_width, big_height, SDL_WINDOW_SHOWN);
                if (!window) {
                    printf("Window creation failed: %c\n", SDL_GetError());
                    RUNNING = false;
                }

                renderer = SDL_CreateRenderer(window, -1, /*0*/SDL_RENDERER_ACCELERATED);
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
            SDL_UpdateTexture(texture, nullptr, big_img_bic, big_width*3);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);

            fps_msg = TTF_RenderText_Solid(Sans, fps_str, White);
            fps_txt = SDL_CreateTextureFromSurface(renderer, fps_msg);

            SDL_RenderCopy(renderer, fps_txt, NULL, &Message_rect);

            SDL_RenderPresent(renderer);

            SDL_PollEvent(&event);
            SDL_DestroyTexture(texture);

            SDL_FreeSurface(fps_msg);
            SDL_DestroyTexture(fps_txt);

            free(img); free(big_img_nn); free(big_img_nn_grey); free(big_img_bic); free(big_img_bic_grey); free(big_img_dif); free(big_img_dif_grey);
            count++;
            current_img++;

            if (current_img > max_image)
                current_img = 1;
        }


        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();

        free(width); free(height);  
    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int naiveCudaExecution()
{
    int* width;
    int* height;

    int const_width;
    int const_height;

    unsigned char* img;

    int big_width;
    int big_height;

    float diff;
    unsigned char* big_img_nn;
    unsigned char* big_img_bic;
    unsigned char* big_img_dif;

    unsigned char* img_cuda;
    unsigned char* big_img_nn_cuda;
    unsigned char* big_img_bic_cuda;
    unsigned char* big_img_nn_grey_cuda;
    unsigned char* big_img_bic_grey_cuda;

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

            cudaDeviceSynchronize();

            cudaStatus = cudaMalloc((void**)&big_img_nn_cuda, big_width * big_height * sizeof(unsigned char) * 3);
            if (cudaStatus != cudaSuccess)
                fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            cudaStatus = cudaMalloc((void**)&big_img_bic_cuda, big_width * big_height * sizeof(unsigned char) * 3);
            if (cudaStatus != cudaSuccess)
                fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            cudaStatus = cudaMalloc((void**)&big_img_nn_grey_cuda, big_width * big_height * sizeof(unsigned char));
            if (cudaStatus != cudaSuccess)
                fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            cudaStatus = cudaMalloc((void**)&big_img_bic_grey_cuda, big_width * big_height * sizeof(unsigned char));
            if (cudaStatus != cudaSuccess)
                fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            if (cudaMalloc((void**)&img_cuda, const_width * const_height * sizeof(unsigned char) * 3) != cudaSuccess)
                printf("Small Image Failed To Copy To Device.\n");      //Notify failure

            cudaMemcpy(img_cuda, img, sizeof(unsigned char) * const_width * const_height * 3, cudaMemcpyHostToDevice);

            dim3 Grid(((big_width - 1) / block_dim) + 1, ((big_height - 1) / block_dim) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
            dim3 Block(block_dim, block_dim);

            //Launch the kernel and pass device matricies and size information
            nearestNeighborsKernel <<< Grid, Block >> > (big_img_nn_cuda, img_cuda, big_width, big_height, const_width, const_height, scale);
            bicubicInterpolationKernel <<< Grid, Block >> > (big_img_nn_cuda, img_cuda, big_width, big_height, const_width, const_height, scale);

            cudaDeviceSynchronize();

            RGB2GreyscaleKernel <<< Grid, Block >>> (big_img_nn_cuda, big_img_nn_grey_cuda, big_width, big_height);
            RGB2GreyscaleKernel <<< Grid, Block >>> (big_img_bic_cuda, big_img_bic_grey_cuda, big_width, big_height);
            
            cudaDeviceSynchronize();

            cudaMemcpy(big_img_nn, big_img_nn_cuda, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);
            cudaMemcpy(big_img_bic, big_img_bic_cuda, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);


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

            SDL_UpdateTexture(texture, nullptr, big_img_nn, big_width * 3);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);

            fps_msg = TTF_RenderText_Solid(Sans, fps_str, White);
            fps_txt = SDL_CreateTextureFromSurface(renderer, fps_msg);

            SDL_RenderCopy(renderer, fps_txt, NULL, &Message_rect);

            SDL_RenderPresent(renderer);

            SDL_PollEvent(&event);
            SDL_DestroyTexture(texture);

            SDL_FreeSurface(fps_msg);
            SDL_DestroyTexture(fps_txt);

            free(img); free(big_img_nn); free(big_img_bic);   //free(big_img_dif);
            cudaFree(img_cuda); cudaFree(big_img_nn_cuda); cudaFree(big_img_bic_cuda);
            cudaFree(big_img_nn_grey_cuda); cudaFree(big_img_bic_grey_cuda);

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

int Code_Testing()
{
    unsigned char* lr_img;
    int lr_width;
    int lr_height;

    unsigned char* hr_img;
    int hr_width;
    int hr_height;

    float scale = 3.0;


    char file_name[50] = "./Testing_Images/image108.ppm";

    



    lr_img = (unsigned char*)readPPM(file_name, &lr_width, &lr_height);

    hr_width  = scale * lr_width;
    hr_height = scale * lr_height;

    //Pointers for each major step
    unsigned char* hr_img_nn            = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height * 3);
    unsigned char* hr_img_nn_grey       = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height);
    unsigned char* hr_img_bic           = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height * 3);
    unsigned char* hr_img_bic_grey      = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height);
    unsigned char* hr_img_diff_grey     = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height);     //Convert to 0-255 unsigned char for image saving
    unsigned char* hr_img_ssim_grey     = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height);     //Convert to 0-255 unsigned char for image saving
    unsigned char* hr_img_artifact_grey = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height);     //Convert to 0-255 unsigned char for image saving
    unsigned char* hr_img_artifact_blurred_grey = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height);     //Convert to 0-255 unsigned char for image savin
    unsigned char* hr_img_fused         = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height * 3);     //Convert to 0-255 unsigned char for image saving
    float* hr_diff_map                  = (float*)malloc(sizeof(float) * hr_width * hr_height);                     //Use for artifact detection
    float* hr_ssim_map                  = (float*)malloc(sizeof(float) * hr_width * hr_height);                     //Use for artifact detection
    float* hr_artifact_map              = (float*)malloc(sizeof(float) * hr_width * hr_height);                     //Use for artifact detection
    float* hr_artifact_blurred_map      = (float*)malloc(sizeof(float) * hr_width * hr_height);                     //Use for artifact detection



    nearestNeighbors(hr_img_nn, hr_width, hr_height, lr_img, lr_width, lr_height, scale);
    RGB2Greyscale(hr_img_nn_grey, hr_img_nn, hr_width, hr_height);
    bicubicInterpolation(hr_img_bic, hr_width, hr_height, lr_img, lr_width, lr_height, scale);
    RGB2Greyscale(hr_img_bic_grey, hr_img_bic, hr_width, hr_height);

    ABS_Difference_Grey(hr_diff_map, hr_img_nn_grey, hr_img_bic_grey, hr_width, hr_height);
    SSIM_Grey(hr_ssim_map, hr_img_nn_grey, hr_img_bic_grey, hr_width, hr_height);
    MapMul(hr_artifact_map, hr_diff_map, hr_ssim_map, hr_width, hr_height);
    
    //MapThreshold(hr_artifact_map, 0.1, hr_width, hr_height);

    //GuassianBlur_Img(hr_img_artifact_blurred_grey, hr_img_bic_grey, hr_width, hr_height, 3, 1.5);
    GuassianBlur_Map(hr_artifact_blurred_map, hr_artifact_map, hr_width, hr_height, 3, 1.5);
    
    MapThreshold(hr_artifact_blurred_map, 0.05, hr_width, hr_height);

    Image_Fusion(hr_img_fused, hr_img_nn, hr_img_bic, hr_artifact_blurred_map, hr_width, hr_height);
    //Image_Fusion(hr_img_fused, hr_img_bic, hr_img_nn, hr_artifact_blurred_map, hr_width, hr_height);


    Map2Greyscale(hr_img_diff_grey, hr_diff_map, hr_width, hr_height, 255);           //Diff values are already between 0-255
    Map2Greyscale(hr_img_ssim_grey, hr_ssim_map, hr_width, hr_height, 255);         //SSIM values are between 0-1 so scale up to 255
    Map2Greyscale(hr_img_artifact_grey, hr_artifact_map, hr_width, hr_height, 255);   //Artifact values should be between 0-255;
    Map2Greyscale(hr_img_artifact_blurred_grey, hr_artifact_blurred_map, hr_width, hr_height, 255);   //Artifact values should be between 0-255;


    writePPM("./Testing_Images/NN.ppm", (char*)hr_img_nn, hr_width, hr_height);
    writePPM("./Testing_Images/BIC.ppm", (char*)hr_img_bic, hr_width, hr_height);
    writePPMGrey("./Testing_Images/NN_Grey.ppm", (char*)hr_img_nn_grey, hr_width, hr_height);
    writePPMGrey("./Testing_Images/BIC_Grey.ppm", (char*)hr_img_bic_grey, hr_width, hr_height);
    writePPMGrey("./Testing_Images/DIFF_Grey.ppm", (char*)hr_img_diff_grey, hr_width, hr_height);
    writePPMGrey("./Testing_Images/SSIM_Grey.ppm", (char*)hr_img_ssim_grey, hr_width, hr_height);
    writePPMGrey("./Testing_Images/Artifact_Grey.ppm", (char*)hr_img_artifact_grey, hr_width, hr_height);
    writePPMGrey("./Testing_Images/Artifact_Grey_Blurred.ppm", (char*)hr_img_artifact_blurred_grey, hr_width, hr_height);
    writePPM("./Testing_Images/FUSED_IMAGE.ppm", (char*)hr_img_fused, hr_width, hr_height);


    //Free memory
    free(hr_img_nn);
    free(hr_img_nn_grey);
    free(hr_img_bic);
    free(hr_img_bic_grey);
    free(hr_img_diff_grey);
    free(hr_img_ssim_grey);
    free(hr_img_artifact_grey);
    free(hr_img_artifact_blurred_grey);
    free(hr_img_fused);

    free(hr_diff_map);
    free(hr_ssim_map);
    free(hr_artifact_map);
    free(hr_artifact_blurred_map);

    return 0;
}

int main()
{
    //return serialExecution();
    //return naiveCudaExecution();
    return Code_Testing();

}