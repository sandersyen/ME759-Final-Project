#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:01:00
#SBATCH -J clahe2
#SBATCH -o out -e err
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
module purge
module load cuda
nvcc main2.cu clahe2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o clahe2
./clahe2