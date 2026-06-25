#!/bin/bash

#SBATCH --job-name=feature_extractor
#SBATCH --output=feature_extractor.out
#SBATCH --error=feature_extractor.err
#SBATCH --nodes=2
#SBATCH --partition=unite
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1

module add unite/nvhpc/25.9

source .venv/bin/activate
mpiexec --bind-to none -np $SLURM_NTASKS python -m mpi4py.futures -m scripts.run_distributed
