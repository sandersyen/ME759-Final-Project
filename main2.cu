// UW ID: cchang253, syeh6
// Name: Chun-Ming Chang, Shang-Yen Yeh

#include <iostream>
#include <string>
#include "clahe2.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BIN_SIZE 101

using namespace std;

int main(int argc, char *argv[])
{
    /*
        the way to run this program should be ./program_name [input_img_name] [output_img_name] [grid_size] [threshold]
    */
    if (argc < 3)
    {
        cout << "Didn't give enough arguments while calling CLAHE editor!\n";
    }
    else
    {
        try
        {
            string inputImg = argv[1];
            string outputImg = argv[2];
            int grid_size = stol(argv[3]);
            int threshold = stol(argv[4]);
            
            int width, height, channel;
            unsigned char* rgb_image = stbi_load(inputImg.c_str(), &width, &height, &channel, 3); // 3 means RGB
            int N = width * height;
            int threads_per_block = grid_size * grid_size;
            int num_block = ((width + grid_size - 1) / grid_size) * ((height + grid_size - 1) / grid_size);

            float* dL;
            float* dA;
            float* dB;
            unsigned char* dImg;

            cudaMallocManaged((void **) &dImg, N*channel*sizeof(unsigned char));
            cudaMallocManaged((void **) &dL, N * sizeof(float));
            cudaMallocManaged((void **) &dA, N * sizeof(float));
            cudaMallocManaged((void **) &dB, N * sizeof(float));

            cudaMemcpy(dImg, rgb_image, N*channel*sizeof(unsigned char), cudaMemcpyHostToDevice);
            cudaMemset(dL, 0.0, N * sizeof(float));
            cudaMemset(dA, 0.0, N * sizeof(float));
            cudaMemset(dB, 0.0, N * sizeof(float));

            // Time the calculation actions exception for read and write image.
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            transformRgbToLab<<<num_block, threads_per_block>>>(dImg, width, height, dL, dA, dB);
            cudaDeviceSynchronize();

            int block_dim = grid_size;
            dim3 dimBlock(block_dim, block_dim);
            dim3 dimGrid((width + dimBlock.x - 1)/dimBlock.x, (height+dimBlock.y -1)/dimBlock.y );

            float* dCdf;
            cudaMallocManaged((void **) &dCdf, num_block * BIN_SIZE * sizeof(float));
            cudaMemset(dCdf, 0.0, num_block * BIN_SIZE * sizeof(float));

            clahe<<<dimGrid, dimBlock>>>(dL, width, height, threshold, dCdf);
            cudaDeviceSynchronize();
            
            pixelInterpolate<<<dimGrid, dimBlock>>>(dL, width, height, dCdf);
            cudaDeviceSynchronize();

            transformLabToRgb<<<num_block, threads_per_block>>>(dImg, width, height, dL, dA, dB);
            cudaDeviceSynchronize();

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            // Get the elapsed time in milliseconds
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            cout << ms << endl;


            cudaMemcpy(rgb_image, dImg, N*channel*sizeof(unsigned char), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            stbi_write_png(outputImg.c_str(), width, height, channel, rgb_image, width*3);

            stbi_image_free(rgb_image);

            cudaFree(dL); cudaFree(dA); cudaFree(dB); cudaFree(dImg);
        }
        catch (...)
        {
            cout << "Invalid argument!\n";
        }
    }

    cout << "\n";
    return 0;
}