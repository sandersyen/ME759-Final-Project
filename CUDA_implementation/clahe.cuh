#ifndef CLAHE_CUH
#define CLAHE_CUH

__global__ void clahe(float* L, int width, int height, int threshold, float* dCdf);

__global__ void transformRgbToLab(unsigned char* pixels, int width, int height, float* L, float* A, float* B);

__global__ void transformLabToRgb(unsigned char* pixels, int width, int height, float* L, float* A, float* B);

__global__ void pixelInterpolate(float* L, int width, int height, float* dCdf);

__device__ void computeHistogram(float* L, int width, int height, int* bins);

__device__ void clipHistogram(int* bins, int threshold);

__device__ void generateCdf(int* bins, float* dCdf);

__device__ void mappingCdf(float* L, int width, int height, float* cdf, float* dCdf); // average neighbor -> mapping

#endif
