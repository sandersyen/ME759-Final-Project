#include <iostream>
#include <cmath>
#include <string>
#include <chrono>
#include <ratio>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

#define BIN_SIZE 101

// globals
int *bin = new int[BIN_SIZE];

void transformRgbToLab(unsigned char *pixels, int width, int height, float *L, float *A, float *B)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = i * width + j;
            float r = (float)pixels[3 * index] / 255;
            float g = (float)pixels[3 * index + 1] / 255;
            float b = (float)pixels[3 * index + 2] / 255;

            r = (r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
            r *= 100;
            g = (g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
            g *= 100;
            b = (b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : b / 12.92;
            b *= 100;

            // reference standard sRGB
            float x = (r * 0.412453 + g * 0.357580 + b * 0.180423) / 95.047;
            float y = (r * 0.212671 + g * 0.715160 + b * 0.072169) / 100.;
            float z = (r * 0.019334 + g * 0.119193 + b * 0.950227) / 108.883;

            x = (x > 0.008856) ? pow(x, 0.3333) : (7.787 * x) + 0.137931;
            y = (y > 0.008856) ? pow(y, 0.3333) : (7.787 * y) + 0.137931;
            z = (z > 0.008856) ? pow(z, 0.3333) : (7.787 * z) + 0.137931;

            L[index] = (116.0 * y) - 16.0;
            A[index] = 500.0 * (x - y);
            B[index] = 200.0 * (y - z);
        }
    }
}

void transformLabToRgb(unsigned char *pixels, int width, int height, float *L, float *A, float *B)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = i * width + j;
            float y = (L[index] + 16.) / 116.;
            float x = A[index] / 500. + y;
            float z = y - B[index] / 200.;
            z = (z > 0.0) ? z : 0.0;

            y = (pow(y, 3) > 0.008856) ? pow(y, 3) : (y - 16. / 116.) / 7.787;
            x = (pow(x, 3) > 0.008856) ? pow(x, 3) : (x - 16. / 116.) / 7.787;
            z = (pow(z, 3) > 0.008856) ? pow(z, 3) : (z - 16. / 116.) / 7.787;

            x = 95.047 * x / 100.;
            y = 100.000 * y / 100.;
            z = 108.883 * z / 100.;

            float r = x * 3.2406 + y * -1.5372 + z * -0.4986;
            float g = x * -0.9689 + y * 1.8758 + z * 0.0415;
            float b = x * 0.0557 + y * -0.2040 + z * 1.0570;

            r = (r > 0.0031308) ? 1.055 * pow(r, (1 / 2.4)) - 0.055 : 12.92 * r;
            g = (g > 0.0031308) ? 1.055 * pow(g, (1 / 2.4)) - 0.055 : 12.92 * g;
            b = (b > 0.0031308) ? 1.055 * pow(b, (1 / 2.4)) - 0.055 : 12.92 * b;

            r = (r > 1) ? 1 : r;
            r = (r > 0) ? r : 0;
            g = (g > 1) ? 1 : g;
            g = (g > 0) ? g : 0;
            b = (b > 1) ? 1 : b;
            b = (b > 0) ? b : 0;

            pixels[3 * index] = (unsigned char)(r * 255.);
            pixels[3 * index + 1] = (unsigned char)(g * 255.);
            pixels[3 * index + 2] = (unsigned char)(b * 255.);
        }
    }
}

static float *histogram(float *image, int width, int height, int r, int c, float threshold, int grid_size)
{
    int sR = r - (grid_size / 2);
    int sC = c - (grid_size / 2);

    float pixVal;
    int a, b, binLoc;
    static int count;

    if (c == 0)
    {
        count = 0; // redundant step, should be removed

        for (int i = 0; i < BIN_SIZE; i++)
            bin[i] = 0;

        for (int i = 0; i < grid_size; i++)
        {
            for (int j = 0; j < grid_size; j++)
            {
                a = sR + i;
                b = sC + j;

                if ((a >= 0 && a < height) && (b >= 0 && b < width))
                {

                    pixVal = image[a * width + b];
                }
                else
                {
                    pixVal = 0.f; // has to be decided
                }

                // alternate logic, takes less iterations
                binLoc = (int)pixVal;

                // to accomodate the max value in the range
                if (binLoc == BIN_SIZE)
                    binLoc--;

                bin[binLoc] += 1;
                count++;
            }
        }
    }
    else
    {
        // remove the old row
        b = sC - 1;
        for (int j = 0; j < grid_size; j++)
        {
            a = sR + j;

            if ((a >= 0 && a < height) && (b >= 0 && b < width))
            {
                pixVal = image[a * width + b];
            }
            else
            {
                pixVal = 0.f; // has to be decided
            }

            binLoc = (int)pixVal;

            // to accomodate the max value in the range
            if (binLoc == BIN_SIZE)
                binLoc--;

            bin[binLoc] -= 1;
        }

        // add the new row
        b = sC + grid_size - 1;
        for (int j = 0; j < grid_size; j++)
        {
            a = sR + j;

            if ((a >= 0 && a < height) && (b >= 0 && b < width))
            {
                pixVal = image[a * width + b];
            }
            else
            {
                pixVal = 0.f; // has to be decided
            }

            binLoc = (int)pixVal;

            // to accomodate the max value in the range
            if (binLoc == BIN_SIZE)
                binLoc--;

            bin[binLoc] += 1;
        }
    }

    float *binFreq = new float[BIN_SIZE];
    float extraAmount = 0.f;

    // calculating the relative frequency
    // and the total extra amount > clip level
    for (int i = 0; i < BIN_SIZE; i++)
    {
        binFreq[i] = ((float)bin[i] / (float)count);
        if (binFreq[i] > threshold)
        {
            extraAmount += (binFreq[i] - threshold);
            binFreq[i] = threshold;
        }
    }

    // adjusting the total extra amount
    if (extraAmount > 0.f)
    {
        float extraAmountPerBin = float(extraAmount / BIN_SIZE);
        for (int i = 0; i < BIN_SIZE; i++)
        {
            binFreq[i] += extraAmountPerBin;
        }
    }

    return binFreq;
}

float *clahe(float *image_in, int width, int height, float threshold, int grid_size)
{
    float *image_out = new float[width * height];

    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            float pixVal = image_in[r * width + c];

            // create the histogram
            float *binFreq = histogram(image_in, width, height, r, c, threshold, grid_size);
            // find the bin location for the center pixel
            int binLoc = (int)pixVal;

            // to accomodate the max value in the range
            if (binLoc == BIN_SIZE)
                binLoc--;

            if (binLoc > (BIN_SIZE - 1))
            {
                cout << "Error in calculating binLoc\n";
                exit(-1);
            }

            float cdf = 0.f;
            for (int k = 0; k <= binLoc; k++)
            {
                cdf += binFreq[k];
            }

            float newPixVal = round(cdf * 100.f);

            image_out[r * width + c] = newPixVal;
        }
    }

    return image_out;
}

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
            float threshold = stof(argv[4]);

            // variables for timing purpose
            high_resolution_clock::time_point start;
            high_resolution_clock::time_point end;
            duration<double, std::milli> duration_sec;

            int width, height, channel;
            unsigned char *rgb_image = stbi_load(inputImg.c_str(), &width, &height, &channel, 3); // 3 means
            cout << inputImg << ": " << width << "x" << height << "x" << channel << "\n";

            float *L = new float[width * height];
            float *A = new float[width * height];
            float *B = new float[width * height];

            start = high_resolution_clock::now(); // Get the starting timestamp

            transformRgbToLab(rgb_image, width, height, L, A, B);

            end = high_resolution_clock::now(); // Get the ending timestamp
            duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
            cout << "transformRgbToLab: " << duration_sec.count() << "ms\n";

            start = high_resolution_clock::now(); // Get the starting timestamp

            float *newL = clahe(L, width, height, threshold, grid_size);

            end = high_resolution_clock::now(); // Get the ending timestamp
            duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
            cout << "clahe: " << duration_sec.count() << "ms\n";

            unsigned char *out_image = new unsigned char[width * height * channel];
            start = high_resolution_clock::now(); // Get the starting timestamp

            transformLabToRgb(out_image, width, height, newL, A, B);

            end = high_resolution_clock::now(); // Get the ending timestamp
            duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
            cout << "transformLabToRgb: " << duration_sec.count() << "ms\n";

            stbi_write_png(outputImg.c_str(), width, height, channel, out_image, width * 3);

            stbi_image_free(rgb_image);
        }
        catch (...)
        {
            cout << "Invalid argument!\n";
        }
    }
    return 0;
}
