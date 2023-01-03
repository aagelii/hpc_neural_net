#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -J project
#SBATCH -o %x.out -e %x.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00

module load nvidia/cuda/11.6.0

# nvcc -lcuda -lcublas *.cu -o CNN  -arch=compute_20 -Wno-deprecated-gpu-targets
nvcc *.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcuda -lcublas -std c++17 -o CNN

echo "Running the code"
./CNN