
const int CHN_NUM = 3;

void Average(float* Avg, float* img_1, float* img_2, int* width, int* height);
void Average(float* Avg, unsigned char* img_1, unsigned char* img_2, int* width, int* height);

void StandardDeviationSquare(float* STD, float* Avg, float* img_1, float* img_2, int* width, int* height);
void StandardDeviationSquare(float* STD, float* Avg, unsigned char* img_1, unsigned char* img_2, int* width, int* height);

void SSIM(float* ssim, float* img_1, float* img_2, int* width, int* height);
void SSIM(float* ssim, unsigned char* img_1, unsigned char* img_2, int* width, int* height);
void SSIM_Grey(float* ssim_map, unsigned char* img_1, unsigned char* img_2, int width, int height);

void ABS_Difference(float* img_diff, float* img_1, float* img_2, int* width, int* height);
void ABS_Difference(unsigned char* img_diff, unsigned char* img_1, unsigned char* img_2, int* width, int* height);
void ABS_Difference_Grey(float* diff_map, unsigned char* img_1, unsigned char* img_2, int width, int height);

void RGB2Greyscale(unsigned char* grey_img, unsigned char* rgb_img, int width, int height);
void Map2Greyscale(unsigned char* grey_img, float* map, int width, int height, int scale);

void MapMul(float* product_map, float* map_1, float* map_2, int width, int height);
void MapThreshold(float* map, float threshold, int width, int height);

void GuassianBlur_Map(float* blur_map, float* input_map, int width, int height, int radius, float sigma);
void GuassianBlur_Img(unsigned char* blur_img, unsigned char* input_img, int width, int height, int radius, float sigma);

void Image_Fusion(unsigned char* fused_img, unsigned char* img_1, unsigned char* img_2, float* weight_map, int width, int height);