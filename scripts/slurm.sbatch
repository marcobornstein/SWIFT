#!/usr/bin/env bash
# SLURM template. Adjust the account, partition and module names for your
# cluster. Create the log directory first -- SLURM opens --output before the
# script runs, so a missing directory fails the job before it starts:
#
#   mkdir -p logs
#   sbatch --ntasks=16 --gres=gpu:4 scripts/slurm.sbatch configs/baseline-ring.yaml
#
# One task per client. Give each node as many GPUs as it has tasks where you
# can; clients otherwise share, which is correct but slower and makes the
# reported times reflect contention rather than the algorithm.
#
#SBATCH --job-name=swift
#SBATCH --time=24:00:00
#SBATCH --ntasks=16
#SBATCH --gres=gpu:4
#SBATCH --mem=32gb
#SBATCH --output=logs/swift-%j.out
set -euo pipefail

module load openmpi cuda || true

CONFIG=${1:?usage: sbatch scripts/slurm.sbatch <config.yaml> [extra flags...]}
shift

# Download the dataset before the job fans out: every task shares a filesystem,
# and concurrent downloads to one path corrupt each other. swift-train also
# serialises this internally, but doing it here keeps the tasks from idling.
python -c "
from pathlib import Path
from swift.data import prepare_dataset
import sys, yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
prepare_dataset(Path(config.get('data_dir', 'data')))
" "$CONFIG"

srun --mpi=pmix swift-train --config "$CONFIG" --download 0 "$@"
