#!/bin/bash -l
#
# Run the study as a SLURM job array, one task per array index, then build
# the table once all tasks are done.  From the repository root:
#
#   N=$(python run_simulations.py --num-tasks)
#   sbatch --array=0-$((N-1)) slurm/run_array.sbatch
#   # when all tasks have finished:
#   python make_table.py
#
# Load R (with dsp) and Python as needed on your cluster, e.g.
#   module load r/4.6.1
#
#SBATCH --job-name=nupspline
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python run_simulations.py --task "$SLURM_ARRAY_TASK_ID"
