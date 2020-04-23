#include <iostream>
#include <stdio.h>
#include "clahe.cuh"

#define BIN_SIZE 101

__global__ void clahe(){} // <<<64, 1024>>>

__global__ void transformRgbToLab(unsigned char* pixels, int width, int height, float* L, float* A, float* B){
    // pixels: length = width * height * 3
    // L: length = width * height
    // A: length = width * height
    // B: length = width * height

    //https://stackoverflow.com/questions/49150250/convert-bgr-to-lab-without-opencv
    //http://www.easyrgb.com/en/math.php

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < width * height)
    {
        float r = (float)pixels[3*i]/255;
        float g = (float)pixels[3*i+1]/255;
        float b = (float)pixels[3*i+2]/255;

        r = (r > 0.04045) ? std::pow((r + 0.055) / 1.055, 2.4) : r / 12.92; r *= 100;
        g = (g > 0.04045) ? std::pow((g + 0.055) / 1.055, 2.4) : g / 12.92; g *= 100;
        b = (b > 0.04045) ? std::pow((b + 0.055) / 1.055, 2.4) : b / 12.92; b *= 100;

        // reference standard sRGB
        float x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 94.811;
        float y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 100.000;
        float z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 107.304;

        x = (x > 0.008856)? std::pow(x, 0.3333) : (7.787 * x) + 0.137931;
        y = (y > 0.008856)? std::pow(y, 0.3333) : (7.787 * y) + 0.137931;
        z = (z > 0.008856)? std::pow(z, 0.3333) : (7.787 * z) + 0.137931;

        L[i] = (116.0 * y) - 16.0;
        A[i] = 500.0 * (x - y);
        B[i] = 200.0 * (y - z);
    }
}
// https://github.com/berendeanicolae/ColorSpace/blob/master/src/Conversion.cpp

__global__ void transformLabToRgb(unsigned char* pixels, int width, int height, float* L, float* A, float* B)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < width * height)
    {
        float y = (L[i] + 16) / 116;
        float x = (A[i]) / 500 + y;
        float z = y - B[i] / 200;

        x = 0.95047 * ((x * x * x > 0.008856) ? x * x * x : (x - 16/116) / 7.787);
        y = 1.00000 * ((y * y * y > 0.008856) ? y * y * y : (y - 16/116) / 7.787);
        z = 1.08883 * ((z * z * z > 0.008856) ? z * z * z : (z - 16/116) / 7.787);

        float r = x *  3.2406 + y * -1.5372 + z * -0.4986;
        float g = x * -0.9689 + y *  1.8758 + z *  0.0415;
        float b = x *  0.0557 + y * -0.2040 + z *  1.0570;

        r = (r > 0.0031308) ? (1.055 * std::pow(r, 1/2.4) - 0.055) : 12.92 * r;
        g = (g > 0.0031308) ? (1.055 * std::pow(g, 1/2.4) - 0.055) : 12.92 * g;
        b = (b > 0.0031308) ? (1.055 * std::pow(b, 1/2.4) - 0.055) : 12.92 * b;

        r = (r > 1.0)? 1.0 : r;
        r = (r < 0.0)? 0.0 : r;
        r = r * 255;

        g = (g > 1.0)? 1.0 : g;
        g = (g < 0.0)? 0.0 : g;
        g = g * 255;

        b = (b > 1.0)? 1.0 : b;
        b = (b < 0.0)? 0.0 : b;
        b = b * 255;

        r = (int) r;
        g = (int) g;
        b = (int) b;

        pixels[3*i] = (unsigned char)r;
        pixels[3*i+1] = (unsigned char)g;
        pixels[3*i+2] = (unsigned char)b;
    }
}

__global__ void computeHistogram(float* L, int width, int height, int* bins)
{
    // L: length = width * height
    // bins: length = BIN_SIZE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < width * height)
    {
        int c = (int)L[i];
        atomicAdd(&bins[c], 1);
    }
}

__global__ void clipHistogram(int* bins, int threshold)
{
    __shared__ int count_overlimit;
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i == 0) count_overlimit=0;
    __syncthreads();

    if(i < BIN_SIZE)
    {
        if(bins[i] > threshold)
        {   
            atomicAdd(&count_overlimit, bins[i] - threshold);
            bins[i] = threshold;
        }
    }
    __syncthreads();
    if(i < BIN_SIZE) bins[i] = bins[i] + count_overlimit/BIN_SIZE + (i < count_overlimit%BIN_SIZE);
}

__global__ void generateCdf(int* bins, float* cdf)
{
    int i = threadIdx.x;
    // small array so here use sequential scan
    if(i==0){
        for(int j = 1; j < BIN_SIZE; j++)
            bins[j]= bins[j]+bins[j-1];
    }
    __syncthreads();
    
    if( i < BIN_SIZE )
        cdf[i] = (float)bins[i]/(float)bins[BIN_SIZE-1];
}

__global__ void mappingCdf(float* L, float* cdf)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    // average neighbor -> mapping
    int index = (int)L[i];
    L[i] = cdf[index] * 100;
}

