#ifndef CLAHE_CUH
#define CLAHE_CUH

__global__ void clahe(); // <<<64, 1024>>>

__global__ void transformRgbToLab(unsigned char* pixels, int width, int height, float* L, float* A, float* B);

__global__ void transformLabToRgb(unsigned char* pixels, int width, int height, float* L, float* A, float* B);

__global__ void computeHistogram(float* L, int width, int height, int* bins);

__global__ void clipHistogram(int* bins, int threshold);

__global__ void generateCdf(int* bins, float* cdf);

__global__ void mappingCdf(float* L, float* cdf); // average neighbor -> mapping

#endif
