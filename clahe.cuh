#ifndef CLAHE_CUH
#define CLAHE_CUH

__global__ void clahe(); // <<<64, 1024>>>

__device__ void transformRgbToLab();

__device__ void transformLabToRgb();

__device__ void computeHistgram();

__device__ void generateCdf();

__device__ void mappingCdf(); // average neighbor -> mapping

#endif
