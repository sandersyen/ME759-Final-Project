#ifndef CLAHE_CUH
#define CLAHE_CUH

__global__ void clahe();

__global__ void transformRgbToLab();

__global__ void transformLabToRgb();

__global__ void computeHistgram();

__global__ void generateCdf();

__global__ void mappingCdf();

#endif
