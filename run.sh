#!/bin/bash

#SBATCH --job-name=feature_descriptor
#SBATCH --output=feature_descriptor.out
#SBATCH --error=feature_descriptor.err
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2

module add unite/nvhpc/25.9

export NUMBA_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1

source .venv/bin/activate
perf stat -r 3 mpiexec --bind-to none -np $SLURM_NTASKS python3 -m mpi4py.futures -m scripts.run_distributed
