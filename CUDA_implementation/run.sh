#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:05:00
#SBATCH -J clahe
#SBATCH -o out -e err
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

nvcc main.cu clahe.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o clahe

echo "Time for CLAHE in ms: gird 32 threshold 10"
echo "image size: 512 x 512"
./clahe ../Test_images/lenna_512.bmp lenna_32.bmp 32 10

echo "image size: 1024 x 1024"
./clahe ../Test_images/man_1024.bmp man_32.bmp 32 10
