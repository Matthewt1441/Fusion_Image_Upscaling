
__global__ void nearestNeighborsKernel(unsigned char* big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale);
__global__ void bicubicInterpolationKernel(unsigned char* big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale);