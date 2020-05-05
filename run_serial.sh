#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:01:00
#SBATCH -J serial
#SBATCH -o serial.out -e serial.err
#SBATCH -c 1
g++ clahe_serial.cpp -Wall -O3 -o serial
./serial lenna.bmp lenna_clahe.bmp 64 0.01
./serial man.bmp man_clahe.bmp 64 0.01
