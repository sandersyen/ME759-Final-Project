__global__ void clahe(); // <<<64, 1024>>>

__device__ void transformRgbToLab(unsigned char* pixels, int width, int height, float* L, float* A, float* B){
    // pixels: length = width * height * 3
    // L: length = width * height
    // A: length = width * height
    // B: length = width * height

    //https://stackoverflow.com/questions/49150250/convert-bgr-to-lab-without-opencv
    //http://www.easyrgb.com/en/math.php

    int i = (threadIdx.x + blockDim.x * blockIdx.x)/3;
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
        float z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 107.304;;

        x = (x > 0.008856)? std::pow(x, 0.3333) : (7.787 * x) + 0.137931;
        y = (y > 0.008856)? std::pow(y, 0.3333) : (7.787 * y) + 0.137931;
        z = (z > 0.008856)? std::pow(z, 0.3333) : (7.787 * z) + 0.137931;

        L[i] = (116.0 * y) - 16.0;
        A[i] = 500.0 * (x - y);
        B[i] = 200.0 * (y - z);
    }
}
// https://github.com/berendeanicolae/ColorSpace/blob/master/src/Conversion.cpp

__device__ void transformLabToRgb();

__device__ void computeHistgram(float* L, int width, int height, int* bins)
{
    // L: length = width * height
    // bins: length = 100
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < width * height)
    {
        int c = (int)L[i];
        atomicAdd(&bins[c], 1);
    }
}

__device__ void clipHistogram(int* bins, int threshold)
{

}

__device__ void generateCdf(int* bins, );

__device__ void mappingCdf(); // average neighbor -> mapping

