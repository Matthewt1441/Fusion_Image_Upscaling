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

#include "serial_code.cuh"
#include "ppm_image.cuh"

#include <chrono>

#include "util.cuh"

#ifdef USE_SDL

#include <SDL.h>
#undef main
#include <SDL_ttf.h>
#undef main


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

        int big_pixel_count = 0;

        unsigned char* hr_img_nn;
        unsigned char* hr_img_nn_grey;
        unsigned char* hr_img_bic;
        unsigned char* hr_img_bic_grey;
        unsigned char* hr_img_diff_grey;
        unsigned char* hr_img_ssim_grey;
        unsigned char* hr_img_artifact_grey;
        unsigned char* hr_img_artifact_blurred_grey;
        unsigned char* hr_img_fused;
        float* hr_diff_map;
        float* hr_ssim_map;
        float* hr_artifact_map;
        float* hr_artifact_blurred_map;

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
            big_pixel_count = big_width * big_height;

            //Pointers for each major step
            hr_img_nn = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
            hr_img_nn_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
            hr_img_bic = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
            hr_img_bic_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
            hr_img_diff_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);     //Convert to 0-255 unsigned char for image saving
            hr_img_ssim_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);     //Convert to 0-255 unsigned char for image saving
            hr_img_artifact_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);     //Convert to 0-255 unsigned char for image saving
            hr_img_artifact_blurred_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);     //Convert to 0-255 unsigned char for image savin
            hr_img_fused = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);     //Convert to 0-255 unsigned char for image saving
            hr_diff_map = (float*)malloc(sizeof(float) * big_pixel_count);                     //Use for artifact detection
            hr_ssim_map = (float*)malloc(sizeof(float) * big_pixel_count);                     //Use for artifact detection
            hr_artifact_map = (float*)malloc(sizeof(float) * big_pixel_count);                     //Use for artifact detection
            hr_artifact_blurred_map = (float*)malloc(sizeof(float) * big_pixel_count);                     //Use for artifact detection
            //unsigned char* big_img_ssim = (unsigned char*)malloc(sizeof(unsigned char) * *big_width * *big_height * 3);

            //printf("Image dimensions: %d x %d\n", *width, *height);
            //printf("Upscale Image dimensions: %d x %d\n", *big_width, *big_height);

            nearestNeighbors(hr_img_nn, big_width, big_height, img, const_width, const_height, scale);
            RGB2Greyscale(hr_img_nn_grey, hr_img_nn, big_width, big_height);
            bicubicInterpolation(hr_img_bic, big_width, big_height, img, const_width, const_height, scale);
            RGB2Greyscale(hr_img_bic_grey, hr_img_bic, big_width, big_height);

            ABS_Difference_Grey(hr_diff_map, hr_img_nn_grey, hr_img_bic_grey, big_width, big_height);
            SSIM_Grey(hr_ssim_map, hr_img_nn_grey, hr_img_bic_grey, big_width, big_height);
            MapMul(hr_artifact_map, hr_diff_map, hr_ssim_map, big_width, big_height);

            GuassianBlur_Map(hr_artifact_blurred_map, hr_artifact_map, big_width, big_height, 3, 1.5);

            MapThreshold(hr_artifact_blurred_map, 0.05, big_width, big_height);

            Image_Fusion(hr_img_fused, hr_img_nn, hr_img_bic, hr_artifact_blurred_map, big_width, big_height);

            auto end = std::chrono::high_resolution_clock::now();
            auto dur = end - start;

            processing_time += std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();

            if (firstImg)
            {
                //writePPMGrey("output_NN_grey.ppm", (char*)big_img_nn_grey, big_width, big_height);
                //writePPMGrey("output_BIC_grey.ppm", (char*)big_img_bic_grey, big_width, big_height);
                ///writePPMGrey("output_DIFF_grey.ppm", (char*)big_img_dif_grey, big_width, big_height);
                //writePPMGrey("output_SSIM_grey.ppm", (char*)big_img_ssim_grey, big_width, big_height);

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
            SDL_UpdateTexture(texture, nullptr, hr_img_fused, big_width * 3);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);

            fps_msg = TTF_RenderText_Solid(Sans, fps_str, White);
            fps_txt = SDL_CreateTextureFromSurface(renderer, fps_msg);

            SDL_RenderCopy(renderer, fps_txt, NULL, &Message_rect);

            SDL_RenderPresent(renderer);

            SDL_PollEvent(&event);
            SDL_DestroyTexture(texture);

            SDL_FreeSurface(fps_msg);
            SDL_DestroyTexture(fps_txt);

            free(img);

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

#else 

int serialExecution()
{

    int width;
    int height;

    int big_width;
    int big_height;
    int big_pixel_count;

    //Host Array Pointers, these should always be unsigned char


    //Kernel Parameters
    int scale = 3;    unsigned char*  h_img;                              //Original Small Input Image
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
    bool RUNNING = true;
    bool firstImg = true;

    //Lets start off with timing one image
    try
    {
        //***** Temp *****//
        char fps_str[50];
        char file_name[50];
        int count = 0;

        double frame_cap = 10;
        sprintf(fps_str, "FPS:%.*f", 3, 0.0);

        int max_image = 200;
        int current_img = 37;

        //***** Temp *****//

        //Read in first image initially to get input width and height.
        //sprintf(file_name, "./LAD/LAD_%d.ppm", current_img);
        sprintf(file_name, "./LM_Frame/image%d.ppm", current_img);
        h_img = (unsigned char*)readPPM(file_name, &width, &height);
        free(h_img);

        //Define big image width and height
        big_width = width * scale; big_height = height * scale;
        big_pixel_count = big_width * big_height;

        //******** Malloc Host Images ********//
        h_big_img_nn                    = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        h_big_img_nn_grey               = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_bic                   = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        h_big_img_bic_grey              = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_DIFF_grey             = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);                 
        h_big_img_SSIM_grey             = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_ARTIFACT_grey         = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_BLURRED_ARTIFACT_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_fused                 = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        h_diff_map                      = (float*)malloc(sizeof(float) * big_pixel_count);
        h_ssim_map                      = (float*)malloc(sizeof(float) * big_pixel_count);
        h_artifact_map                  = (float*)malloc(sizeof(float) * big_pixel_count);
        h_blurred_artifact_map          = (float*)malloc(sizeof(float) * big_pixel_count);
        //******** Malloc Host Images ********//

         //Variables for timing
        double processing_time = 0;

        //**************** Run & Time Kernels ****************//
        //auto start = std::chrono::high_resolution_clock::now();

        //Load Input Image
        h_img = (unsigned char*)readPPM(file_name, &width, &height);

        nearestNeighbors(h_big_img_nn, big_width, big_height, h_img, width, height, scale);
        RGB2Greyscale(h_big_img_nn_grey, h_big_img_nn, big_width, big_height);
        //auto end = std::chrono::high_resolution_clock::now();
        bicubicInterpolation(h_big_img_bic, big_width, big_height, h_img, width, height, scale);
        RGB2Greyscale(h_big_img_bic_grey, h_big_img_bic, big_width, big_height);

        ABS_Difference_Grey(h_diff_map, h_big_img_nn_grey, h_big_img_bic_grey, big_width, big_height);
        SSIM_Grey(h_ssim_map, h_big_img_nn_grey, h_big_img_bic_grey, big_width, big_height);
        MapMul(h_artifact_map, h_diff_map, h_ssim_map, big_width, big_height);

        
        auto start = std::chrono::high_resolution_clock::now();

        GuassianBlur_Map(h_blurred_artifact_map, h_artifact_map, big_width, big_height, 3, 1.5);
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = end - start;

        MapThreshold(h_blurred_artifact_map, 0.05, big_width, big_height);

        auto end = std::chrono::high_resolution_clock::now();
        auto dur = end - start;


        Image_Fusion(h_big_img_fused, h_big_img_nn, h_big_img_bic, h_blurred_artifact_map, big_width, big_height);

        //auto end = std::chrono::high_resolution_clock::now();
        //auto dur = end - start;
        processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
        printf("Total compute time (ms) %f\n", processing_time);
        //**************** Run & Time Kernels ****************//

        //Convert Maps to Greyscale images
        Map2Greyscale(h_big_img_DIFF_grey               , h_diff_map                , big_width , big_height, 255); //Diff values are already between 0-255
        Map2Greyscale(h_big_img_SSIM_grey               , h_ssim_map                , big_width , big_height, 255); //SSIM values are between 0-1 so scale up to 255
        Map2Greyscale(h_big_img_ARTIFACT_grey           , h_artifact_map            , big_width , big_height, 255); //Artifact values should be between 0-255;
        Map2Greyscale(h_big_img_BLURRED_ARTIFACT_grey   , h_blurred_artifact_map    , big_width , big_height, 255); //Artifact values should be between 0-255;


        //Save Images
        writePPM    ("./Serial_Output/NN.ppm"                   , (char*)h_big_img_nn                       , big_width, big_height);
        writePPMGrey("./Serial_Output/NN_Grey.ppm"              , (char*)h_big_img_nn_grey                  , big_width, big_height);
        writePPM    ("./Serial_Output/BIC.ppm"                  , (char*)h_big_img_bic                      , big_width, big_height);
        writePPMGrey("./Serial_Output/BIC_Grey.ppm"             , (char*)h_big_img_bic_grey                 , big_width, big_height);
        writePPMGrey("./Serial_Output/DIFF_Grey.ppm"            , (char*)h_big_img_DIFF_grey                , big_width, big_height);
        writePPMGrey("./Serial_Output/SSIM_Grey.ppm"            , (char*)h_big_img_SSIM_grey                , big_width, big_height);
        writePPMGrey("./Serial_Output/ARTIFACT_Grey.ppm"        , (char*)h_big_img_ARTIFACT_grey            , big_width, big_height);
        writePPMGrey("./Serial_Output/BLURRED_ARTIFACT_Grey.ppm", (char*)h_big_img_BLURRED_ARTIFACT_grey    , big_width, big_height);
        writePPM    ("./Serial_Output/FUSED.ppm"                , (char*)h_big_img_fused                    , big_width, big_height);



        //Free Host Memory
        free(h_big_img_nn);             
        free(h_big_img_nn_grey);
        free(h_big_img_bic);            
        free(h_big_img_bic_grey);
        free(h_big_img_DIFF_grey);           
        free(h_big_img_SSIM_grey);          
        free(h_big_img_ARTIFACT_grey);        
        free(h_big_img_BLURRED_ARTIFACT_grey);
        free(h_big_img_fused);
        free(h_diff_map);              
        free(h_ssim_map);                   
        free(h_artifact_map);                 
        free(h_blurred_artifact_map);


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

            img = (unsigned char*)readPPM(file_name, width, height);

            auto start = std::chrono::high_resolution_clock::now();

            const_width = *width;
            const_height = *height;

            big_width = const_width * scale; big_height = const_height * scale;
            big_pixel_count = big_width * big_height;

            //Pointers for each major step
            hr_img_nn = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
            hr_img_nn_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
            hr_img_bic = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
            hr_img_bic_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
            hr_img_diff_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);     //Convert to 0-255 unsigned char for image saving
            hr_img_ssim_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);     //Convert to 0-255 unsigned char for image saving
            hr_img_artifact_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);     //Convert to 0-255 unsigned char for image saving
            hr_img_artifact_blurred_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);     //Convert to 0-255 unsigned char for image savin
            hr_img_fused = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);     //Convert to 0-255 unsigned char for image saving
            hr_diff_map = (float*)malloc(sizeof(float) * big_pixel_count);                     //Use for artifact detection
            hr_ssim_map = (float*)malloc(sizeof(float) * big_pixel_count);                     //Use for artifact detection
            hr_artifact_map = (float*)malloc(sizeof(float) * big_pixel_count);                     //Use for artifact detection
            hr_artifact_blurred_map = (float*)malloc(sizeof(float) * big_pixel_count);                     //Use for artifact detection

            nearestNeighbors(hr_img_nn, big_width, big_height, img, const_width, const_height, scale);
            RGB2Greyscale(hr_img_nn_grey, hr_img_nn, big_width, big_height);
            bicubicInterpolation(hr_img_bic, big_width, big_height, img, const_width, const_height, scale);
            RGB2Greyscale(hr_img_bic_grey, hr_img_bic, big_width, big_height);

            ABS_Difference_Grey(hr_diff_map, hr_img_nn_grey, hr_img_bic_grey, big_width, big_height);
            SSIM_Grey(hr_ssim_map, hr_img_nn_grey, hr_img_bic_grey, big_width, big_height);
            MapMul(hr_artifact_map, hr_diff_map, hr_ssim_map, big_width, big_height);

            GuassianBlur_Map(hr_artifact_blurred_map, hr_artifact_map, big_width, big_height, 3, 1.5);

            MapThreshold(hr_artifact_blurred_map, 0.05, big_width, big_height);

            Image_Fusion(hr_img_fused, hr_img_nn, hr_img_bic, hr_artifact_blurred_map, big_width, big_height);

            auto end = std::chrono::high_resolution_clock::now();
            auto dur = end - start;

            processing_time += std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();

            free(img);

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


            count++;
            current_img++;

            if (current_img > max_image)
                current_img = 1;
        }
#endif

        //free(width); free(height);
    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

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

    hr_width = scale * lr_width;
    hr_height = scale * lr_height;

    //Pointers for each major step
    unsigned char* hr_img_nn = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height * 3);
    unsigned char* hr_img_nn_grey = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height);
    unsigned char* hr_img_bic = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height * 3);
    unsigned char* hr_img_bic_grey = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height);
    unsigned char* hr_img_diff_grey = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height);     //Convert to 0-255 unsigned char for image saving
    unsigned char* hr_img_ssim_grey = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height);     //Convert to 0-255 unsigned char for image saving
    unsigned char* hr_img_artifact_grey = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height);     //Convert to 0-255 unsigned char for image saving
    unsigned char* hr_img_artifact_blurred_grey = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height);     //Convert to 0-255 unsigned char for image savin
    unsigned char* hr_img_fused = (unsigned char*)malloc(sizeof(unsigned char) * hr_width * hr_height * 3);                 //Convert to 0-255 unsigned char for image saving
    float* hr_diff_map = (float*)malloc(sizeof(float) * hr_width * hr_height);                                              //Use for artifact detection
    float* hr_ssim_map = (float*)malloc(sizeof(float) * hr_width * hr_height);                     //Use for artifact detection
    float* hr_artifact_map = (float*)malloc(sizeof(float) * hr_width * hr_height);                     //Use for artifact detection
    float* hr_artifact_blurred_map = (float*)malloc(sizeof(float) * hr_width * hr_height);                     //Use for artifact detection

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
#endif