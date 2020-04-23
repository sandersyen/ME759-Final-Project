// UW ID: cchang253, syeh6
// Name: Chun-Ming Chang, Shang-Yen Yeh

#include <iostream>
#include <cmath>
#include <stdint.h>
#include "clahe.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

__host__ void test_run(){
    int threshold = 50;
    int width, height, channel;
    unsigned char* rgb_image = stbi_load("tsukuba_L.png", &width, &height, &channel, 3); // 3 means RGB
    int N = width * height;
    int threads_per_block = 1024;
    int num_block = (N + threads_per_block-1)/threads_per_block;

    float* dL;
    float* dA;
    float* dB;
    float* L = new float[N];
    float* A = new float[N];
    float* B = new float[N];
    unsigned char* dImg;

    cudaMallocManaged((void **) &dImg, N*channel*sizeof(unsigned char));
    cudaMallocManaged((void **) &dL, N * sizeof(float));
    cudaMallocManaged((void **) &dA, N * sizeof(float));
    cudaMallocManaged((void **) &dB, N * sizeof(float));

    cudaMemcpy(dImg, rgb_image, N*channel*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(dL, 0.0, N * sizeof(float));
    cudaMemset(dA, 0.0, N * sizeof(float));
    cudaMemset(dB, 0.0, N * sizeof(float));

    transformRgbToLab<<<num_block, threads_per_block>>>(dImg, width, height, dL, dA, dB);
    cudaDeviceSynchronize();

    // cudaMemcpy(dB, bins, BIN_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(L, dL, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(A, dA, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, dB, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(int i = 0; i < 3; i++)
        cout << L[i] << "," << A[i] << "," << B[i] << "\n";
        
    for(int i = 0; i < 3; i++)
        cout << L[N-i-1] << "," << A[N-i-1] << "," << B[N-i-1] << "\n";

    int block_dim = 32;
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((width + dimBlock.x - 1)/dimBlock.x, (height+dimBlock.y -1)/dimBlock.y );
    clahe<<<dimGrid, dimBlock>>>(dL, width, height, threshold);
    cudaDeviceSynchronize();
    
    cudaMemcpy(L, dL, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(int i = 0; i < 3; i++)
        cout << L[i] << "," << A[i] << "," << B[i] << "\n";
        
    for(int i = 0; i < 3; i++)
        cout << L[N-i-1] << "," << A[N-i-1] << "," << B[N-i-1] << "\n";

    for(int i = 0; i < N; i++){
        if(L[i] < 0) cout << "L["<< i << "]<0\n";
        if(L[i] > 100) cout << "L["<< i << "]>100\n";
    }

    transformLabToRgb<<<num_block, threads_per_block>>>(dImg, width, height, dL, dA, dB);
    cudaDeviceSynchronize();

    // cudaMemcpy(dB, bins, BIN_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(L, dL, N * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(A, dA, N * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(B, dB, N * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    cudaMemcpy(rgb_image, dImg, N*channel*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_png("image2.png", width, height, channel, rgb_image, width*3);

    stbi_image_free(rgb_image);

    cudaFree(dL); cudaFree(dA); cudaFree(dB); cudaFree(dImg);
}

// __host__ void test_transform() {
//     int width, height, channel;

//     unsigned char* rgb_image = stbi_load("tsukuba_L.png", &width, &height, &channel, 3); // 3 means RGB
//     int N = width * height;
//     int threads_per_block = 1024;
//     int num_block = (N + threads_per_block-1)/threads_per_block;

//     float* dL;
//     float* dA;
//     float* dB;
//     float* L = new float[N];
//     float* A = new float[N];
//     float* B = new float[N];
//     unsigned char* dImg;

//     cudaMallocManaged((void **) &dImg, N*channel*sizeof(unsigned char));
//     cudaMallocManaged((void **) &dL, N * sizeof(float));
//     cudaMallocManaged((void **) &dA, N * sizeof(float));
//     cudaMallocManaged((void **) &dB, N * sizeof(float));

//     cudaMemcpy(dImg, rgb_image, N*channel*sizeof(unsigned char), cudaMemcpyHostToDevice);
//     cudaMemset(dL, 0.0, N * sizeof(float));
//     cudaMemset(dA, 0.0, N * sizeof(float));
//     cudaMemset(dB, 0.0, N * sizeof(float));

//     transformRgbToLab<<<num_block, threads_per_block>>>(dImg, width, height, dL, dA, dB);
//     cudaDeviceSynchronize();

//     // cudaMemcpy(dB, bins, BIN_SIZE * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(L, dL, N * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(A, dA, N * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(B, dB, N * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();

//     for(int i = 0; i < 3; i++){
//         cout << L[i] << "," << A[i] << "," << B[i] << "\n";
//         cout << L[N-i-1] << "," << A[N-i-1] << "," << B[N-i-1] << "\n";
//     }

//     transformLabToRgb<<<num_block, threads_per_block>>>(dImg, width, height, dL, dA, dB);
//     cudaDeviceSynchronize();

//     // cudaMemcpy(dB, bins, BIN_SIZE * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(L, dL, N * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(A, dA, N * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(B, dB, N * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();

//     cudaMemcpy(rgb_image, dImg, N*channel*sizeof(unsigned char), cudaMemcpyDeviceToHost);

//     stbi_write_png("image2.png", width, height, channel, rgb_image, width*3);

//     stbi_image_free(rgb_image);

//     cudaFree(dL); cudaFree(dA); cudaFree(dB); cudaFree(dImg);
// }
// __host__ void test_computeHistogram(){
//     int width = 32, height=32;
//     int BIN_SIZE = 101;
//     int threads_per_block = 1024;

//     float * L = new float[threads_per_block];
//     for(int i =0; i < threads_per_block; i++) L[i] = 5;

//     int *bins = new int[BIN_SIZE];
//     for (int i = 0; i < BIN_SIZE; i++)
//         bins[i] = 0;

//     int* dB;
//     float* dL;
//     int* out = new int[BIN_SIZE];

//     cudaMallocManaged((void **) &dB, BIN_SIZE * sizeof(int));
//     cudaMallocManaged((void **) &dL, width * height * sizeof(float));

//     cudaMemcpy(dB, bins, BIN_SIZE * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(dL, L, width * height * sizeof(float), cudaMemcpyHostToDevice);

//     computeHistogram<<<1, threads_per_block>>>(dL, width, height, dB);
//     cudaMemcpy(out, dB, BIN_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

//     cudaDeviceSynchronize();
//     for(int i = 0; i < BIN_SIZE; i++){
//         if(i==5 && out[i]!=1024) cout << out[i] << "i=5, error\n";
//         if(i!=5 && out[i]!=0) cout << out[i] << "error\n";
//     }

//     cudaFree(dB); cudaFree(dL);
// }

// __host__ void test_func() {
//     int width = 32, height=32;
//     int BIN_SIZE = 101;
//     int threads_per_block = 1024;
//     // int num_block = (N + threads_per_block-1)/threads_per_block;

//     float * L = new float[threads_per_block];
//     for(int i =0; i < threads_per_block; i++) L[i] = 5;

//     int *bins = new int[BIN_SIZE];
//     for (int i = 0; i < BIN_SIZE; i++)
//         bins[i] = 0;

//     int* dB;
//     float* dL;
//     int* out = new int[BIN_SIZE];

//     float* dC;
//     float* cdf = new float[BIN_SIZE];

//     cudaMallocManaged((void **) &dB, BIN_SIZE * sizeof(int));
//     cudaMallocManaged((void **) &dL, width * height * sizeof(float));

//     cudaMemcpy(dB, bins, BIN_SIZE * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(dL, L, width * height * sizeof(float), cudaMemcpyHostToDevice);

//     cudaMallocManaged((void **) &dC, BIN_SIZE * sizeof(float));
//     cudaMemset(dC, 0.0, BIN_SIZE * sizeof(float));

//     // test compute Histogram
//     computeHistogram<<<1, threads_per_block>>>(dL, width, height, dB);
//     cudaMemcpy(out, dB, BIN_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();

//     for(int i = 0; i < BIN_SIZE; i++){
//         if(i==5 && out[i]!=1024) cout << i << "Hist error\n";
//         if(i!=5 && out[i]!=0)    cout << i << "Hist error\n";
//     }

//     // test clipHistogram
//     clipHistogram<<<1, threads_per_block>>>( dB , 50 );
//     cudaMemcpy(out, dB, BIN_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();
    
//     for(int i = 0; i < BIN_SIZE; i++){
//         int value = (1024 - 50)/BIN_SIZE + (i < (1024-50)%BIN_SIZE);
//         if(i==5 && out[i] != 50 + value) cout << i << "Clip error\n";
//         if(i!=5 && out[i] != value)      cout << i << "Clip error\n";
//     }
    
//     // test generateCdf
//     generateCdf<<<1, threads_per_block>>>(dB, dC);
//     cudaMemcpy(cdf, dC, BIN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();
//     float sum = 0;
//     for(int i = 0; i < BIN_SIZE; i++){
//         int value = (1024 - 50)/BIN_SIZE + (i < (1024-50)%BIN_SIZE);
//         if(i==5) value += 50;
//         sum += value;
//         float fvalue = sum/1024.0;
//         if(cdf[i] != fvalue) cout << fvalue <<","<<cdf[i] << "," << i << "CDF error\n";
//     }

//     // test mappingCdf
//     mappingCdf<<<1, threads_per_block>>>(dL, width, height, dC);
//     cudaMemcpy(out, dB, BIN_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

//     float* Lout = new float[width*height];
//     cudaMemcpy(Lout, dL, width * height * sizeof(float), cudaMemcpyDeviceToHost);
//     // average neighbor -> mapping
//     for(int i = 0; i < width*height; i++){
//         int index = L[i];
//         float value = cdf[index] * 100;
//         if(Lout[i] != value) cout << value << "," << Lout[i] << "Mapping error\n";
//     }
    
//     cudaFree(dB); cudaFree(dC); cudaFree(dL);
// }

int main(int argc, char *argv[])
{
    /*
        need to decide how many arguments we need 
    */
    if (argc < 3)
    {
        cout << "Didn't give enough arguments while calling CLAHE editor!\n";
    }
    else
    {
        try
        {
            const int grid_size = 64; // 8 * 8
            const int hyperparemeter = 100; // for histogram
            /*
            

                parse arguments 
                img threshold
            */

            /*
                call kernal functions

                kernal: transform colorspace -> compute CDF -> transform colorspace back
            */

            /*
                out impage
            */
        }
        catch (...)
        {
            cout << "Invalid argument!\n";
        }
    }

    // int width, height, channel;

    // unsigned char* rgb_image = stbi_load("tsukuba_L.png", &width, &height, &channel, 3); // 3 means RGB

    // cout << "Image dimension: (" << width << "," << height << "," << channel << ")\n";
    // cout << (float)rgb_image[0] <<  "," << rgb_image[1] << "," << rgb_image[2]  << "\n";

    // stbi_write_png("image2.png", width, height, channel, rgb_image, width*3);

    // stbi_image_free(rgb_image);
    
    // test_func();
    // test_computeHistogram();
    // test_transform();
    test_run();
    cout << "\n";
    return 0;
}