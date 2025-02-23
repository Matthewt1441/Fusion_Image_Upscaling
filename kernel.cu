
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <string>

#include "helper.cuh"

#include <ctime>

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
        int* window_size = (int*)malloc(sizeof(int));
        *window_size = 8;

        int scale = 2;

        //*width = 320;
        //*height = 240;
        //img = createImage("test.ppm", width, height);

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

        float diff = 0;
        unsigned char* big_img_nn;
        unsigned char* big_img_bic;
        unsigned char* big_img_dif;

        std::time_t start, end;
        
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

        start = std::time(0);

        double frame_cap = 10;
        sprintf(fps_str, "FPS:%.*f", 3, 0.0);

        int max_image = 200;
        int current_img = 1;

        double processing_time = 0;

        while(RUNNING && event.type != SDL_QUIT)
        {
            if (count == frame_cap)
            {
        
                diff = frame_cap / processing_time;
                sprintf(fps_str, "FPS:%.*f", 3, diff);
                
                count = 0;
                processing_time = 0;
            }
            
            sprintf(file_name, "./LM_Frame/image%d.ppm", current_img);

            img = (unsigned char*)readPPM(file_name, width, height);

            start = std::time(0);
            *big_width = *width * scale; *big_height = *height * scale;
            big_img_nn = (unsigned char*)malloc(sizeof(unsigned char) * *big_width * *big_height * 3);
            big_img_bic = (unsigned char*)malloc(sizeof(unsigned char) * *big_width * *big_height * 3);
            big_img_dif = (unsigned char*)malloc(sizeof(unsigned char) * *big_width * *big_height * 3);
            //unsigned char* big_img_ssim = (unsigned char*)malloc(sizeof(unsigned char) * *big_width * *big_height * 3);

            //printf("Image dimensions: %d x %d\n", *width, *height);
            //printf("Upscale Image dimensions: %d x %d\n", *big_width, *big_height);

            nearestNeighbors(big_img_nn, big_width, big_height, img, width, height, scale);
            //nearestNeighbors(big_img_bic, big_width, big_height, img, width, height, scale);
            bicubicInterpolation(big_img_bic, big_width, big_height, img, width, height, scale);
            //ABS_Difference(big_img_dif, big_img_nn, big_img_bic, big_width, big_height);
            Artifact_Detection(big_img_dif, big_img_nn, big_img_bic, big_width, big_height, window_size, 0.9);

            //writePPM("output_NN.ppm", (char*)big_img_nn, big_width, big_height);
            //writePPM("output_BIC.ppm", (char*)big_img_bic, big_width, big_height);
            //writePPM("output_diff.ppm", (char*)big_img_dif, big_width, big_height);

            end = std::time(0);

            processing_time += std::difftime(end, start);

            if (firstImg)
            {
                window = SDL_CreateWindow("PPM Image", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, *big_width, *big_height, SDL_WINDOW_SHOWN);
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

            texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STATIC, *big_width, *big_height);
            if (!texture)
            {
                printf("Texture creation failed: %c \n", SDL_GetError());
                RUNNING = false;
            }

            SDL_UpdateTexture(texture, nullptr, big_img_bic, *big_width * 3);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);

            fps_msg = TTF_RenderText_Solid(Sans, fps_str, White);
            fps_txt = SDL_CreateTextureFromSurface(renderer, fps_msg);

            SDL_RenderCopy(renderer, fps_txt, NULL, &Message_rect);

            SDL_RenderPresent(renderer);

            SDL_PollEvent(&event);
            SDL_DestroyTexture(texture);

            SDL_FreeSurface(fps_msg);
            SDL_DestroyTexture(fps_txt);

            free(img); free(big_img_bic);// free(big_img_nn);  free(big_img_dif);
            count++;
            current_img++;

            if (current_img > max_image)
                current_img = 1;
        }

        
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
      
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