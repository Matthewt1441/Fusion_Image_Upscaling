void nearestNeighbors(unsigned char* big_img_data, int big_width, int big_height, unsigned char* img_data, int width, int height, int scale);
float cubicInterpolate(float p[4], float x);
float bicubicInterpolate(float p[4][4], float x, float y);
void bicubicInterpolation(unsigned char* big_img_data, int big_width, int big_height, unsigned char* img_data, int width, int height, int scale);

void SSIM_Grey(float* ssim_map, unsigned char* img_1, unsigned char* img_2, int width, int height);
void ABS_Difference_Grey(float* diff_map, unsigned char* img_1, unsigned char* img_2, int width, int height);

void RGB2Greyscale(unsigned char* grey_img, unsigned char* rgb_img, int width, int height);
void Map2Greyscale(unsigned char* grey_img, float* map, int width, int height, int scale);

void MapMul(float* product_map, float* map_1, float* map_2, int width, int height);
void MapThreshold(float* map, float threshold, int width, int height);

void GuassianBlur_Map(float* blur_map, float* input_map, int width, int height, int radius, float sigma);
void GuassianBlur_Img(unsigned char* blur_img, unsigned char* input_img, int width, int height, int radius, float sigma);

void Image_Fusion(unsigned char* fused_img, unsigned char* img_1, unsigned char* img_2, float* weight_map, int width, int height);