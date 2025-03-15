
const int CHN_NUM = 3;

void Average(float* Avg, float* img_1, float* img_2, int* width, int* height);
void Average(float* Avg, unsigned char* img_1, unsigned char* img_2, int* width, int* height);

void StandardDeviationSquare(float* STD, float* Avg, float* img_1, float* img_2, int* width, int* height);
void StandardDeviationSquare(float* STD, float* Avg, unsigned char* img_1, unsigned char* img_2, int* width, int* height);

void SSIM(float* ssim, float* img_1, float* img_2, int* width, int* height);
void SSIM(float* ssim, unsigned char* img_1, unsigned char* img_2, int* width, int* height);
void SSIM_Grey(unsigned char* ssim_map, unsigned char* img_1, unsigned char* img_2, int width, int height);

void ABS_Difference(float* img_diff, float* img_1, float* img_2, int* width, int* height);
void ABS_Difference(unsigned char* img_diff, unsigned char* img_1, unsigned char* img_2, int* width, int* height);
void ABS_Difference_Grey(unsigned char* img_diff, unsigned char* img_1, unsigned char* img_2, int width, int height);

void RGB2Greyscale(unsigned char* rgb_img, unsigned char* grey_img, int width, int height);

void WeightMap_Grey(unsigned char* weight_map, unsigned char* img_1, unsigned char* img_2, int width, int height);
