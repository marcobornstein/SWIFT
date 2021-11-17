#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=matcha     # sets the job name if not set from environment
#SBATCH --time=02:30:00                  # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger                  # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                                         # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:2
#SBATCH --ntasks=8
#SBATCH --mem 64gb                                              # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END                 # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE,
#SBATCH --error Logs/%x_%A_%a.log

#module load python
module load openmpi

mpirun -np 8 python train_mpi.py --description Matcha --resSize 50 --randomSeed 9001 --datasetRoot ./data --budget 0.5 --outputFolder Output --bs 64 --epoch 1 --name trial
