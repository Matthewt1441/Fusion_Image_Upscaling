char* readPPM(char* filename, int* width, int* height);
void writePPM(char* filename, char* img_data, int width, int height);
void writePPMGrey(char* filename, char* img_data, int width, int height);
char* createImage(char* filename, int width, int height);