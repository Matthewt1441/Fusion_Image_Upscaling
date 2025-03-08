void nearestNeighbors(unsigned char* big_img_data, int big_width, int big_height, unsigned char* img_data, int width, int height, int scale);
float cubicInterpolate(float p[4], float x);
float bicubicInterpolate(float p[4][4], float x, float y);
void bicubicInterpolation(unsigned char* big_img_data, int big_width, int big_height, unsigned char* img_data, int width, int height, int scale);