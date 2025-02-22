
const int CHN_NUM = 3;

void Average(float* Avg, float* img_1, float* img_2, int* width, int* height);
void Average(float* Avg, unsigned char* img_1, unsigned char* img_2, int* width, int* height);

void StandardDeviationSquare(float* STD, float* Avg, float* img_1, float* img_2, int* width, int* height);
void StandardDeviationSquare(float* STD, float* Avg, unsigned char* img_1, unsigned char* img_2, int* width, int* height);

void SSIM(float* ssim, float* img_1, float* img_2, int* width, int* height);
void SSIM(float* ssim, unsigned char* img_1, unsigned char* img_2, int* width, int* height);

void ABS_Difference(float* img_diff, float* img_1, float* img_2, int* width, int* height);
void ABS_Difference(unsigned char* img_diff, unsigned char* img_1, unsigned char* img_2, int* width, int* height);