// UW ID: cchang253, syeh6
// Name: Chun-Ming Chang, Shang-Yen Yeh

#include <iostream>
#include <cmath>
#include "mmul.h"

using namespace std;

int main(int argc, char *argv[])
{
    /*
        need to decide how many arguments we need 
    */
    if (argc < 3)
    {
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