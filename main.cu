// UW ID: cchang253, syeh6
// Name: Chun-Ming Chang, Shang-Yen Yeh

#include <iostream>
#include <cmath>
#include <stdint.h>
#include <cub/cub.cuh>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

int main(int argc, char *argv[])
{
    /*
        need to decide how many arguments we need 
    */
    if (argc < 3)
    {
        int width, height, channel;

        unsigned char* rgb_image = stbi_load("tsukuba_L.png", &width, &height, &channel, 3); // 3 means RGB

        cout << "Image dimension: (" << width << "," << height << "," << channel << ")\n";

        cout << (float)rgb_image[0] <<  "," << rgb_image[1] << "," << rgb_image[2]  << "\n";

        stbi_write_png("image2.png", width, height, channel, rgb_image, width*3);

        stbi_image_free(rgb_image);

        cout << "Didn't give enough arguments while calling CLAHE editor!";
    }
    else
    {
        try
        {
            
            /*
                grid size 8x8
                hyperparemeter 100
                
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
            cout << "Invalid argument!";
        }
    }

    cout << "\n";
    return 0;
}