__global__ void RGB2GreyscaleKernel(unsigned char* rgb_img, unsigned char* grey_img, int width, int height);
__global__ void nearestNeighborsKernel(unsigned char* big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale);
__global__ void bicubicInterpolationKernel(unsigned char* big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale);
__global__ void ABS_Difference_Grey_Kernel(float* diff_map, unsigned char* img_1, unsigned char* img_2, int width, int height);
__global__ void SSIM_Grey_Kernel(float* ssim_map, unsigned char* img_1, unsigned char* img_2, int width, int height);
__global__ void MapMulKernel(float* product_map, float* map_1, float* map_2, int width, int height);
__global__ void GuassianBlur_Map_Kernel(float* blur_map, float* input_map, int width, int height, int radius, float sigma);
__global__ void MapThreshold_Kernel(float* map, float threshold, int width, int height);
__global__ void Image_Fusion_Kernel(unsigned char* fused_img, unsigned char* img_1, unsigned char* img_2, float* weight_map, int width, int height);