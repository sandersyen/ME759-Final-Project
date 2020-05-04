#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:05:00
#SBATCH -J clahe2
#SBATCH -o out -e err
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

nvcc main2.cu clahe2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o clahe2

echo "Time for CLAHE in ms: gird 32 threshold 10"
echo "image size: 512 x 512"
nvprof --unified-memory-profiling off ./clahe2 lenna.bmp lenna_32.bmp 32 10

echo "image size: 1024 x 1024"
nvprof --unified-memory-profiling off ./clahe2 man.bmp man_32.bmp 32 10
