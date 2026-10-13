#!/bin/bash
#SBATCH --job-name=run_drm_SSFI
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=16
#SBATCH --mem=0
#SBATCH --output=log_run_drm_SSFI.log
pwd; hostname; date
SECONDS=0

source /path/to/venv/bin/activate      # the environment with ShakerMaker

export HDF5_USE_FILE_LOCKING=FALSE

mpirun python -s surface_SSFI.py

echo "Elapsed: $SECONDS seconds."
date