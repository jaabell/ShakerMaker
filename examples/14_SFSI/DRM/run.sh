#!/bin/bash
#SBATCH --job-name=run_drm_SSFI
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=16
#SBATCH --mem=0
#SBATCH --output=log_run_drm_SSFI.log
pwd; hostname; date
SECONDS=0

source ~/v_ENV/clark_kent/bin/activate

export HDF5_USE_FILE_LOCKING=FALSE

/opt/openmpi/bin/mpirun /mnt/deadmanschest/pxpalacios/v_ENV/clark_kent/bin/python -s \drm.py

echo "Elapsed: $SECONDS seconds."
date