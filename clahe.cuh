#ifndef CLAHE_CUH
#define CLAHE_CUH

__global__ void clahe(); // <<<length / 32 * width / 32 , 32 * 32>>>

__device__ void transformRgbToLab();

__device__ void transformLabToRgb();

__device__ void computeHistgram();

__device__ void clipHistgram();

__device__ void redistributeHistgram();

__device__ void generateCdf();

__device__ void mappingCdf(); // average neighbor -> mapping

#endif
