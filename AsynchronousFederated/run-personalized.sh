#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=AsyncDecentralized     # sets the job name if not set from environment
#SBATCH --time=00:09:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=furongh    # set QOS, this will determine what resources can be requested
#SBATCH --qos=medium    # set QOS, this will determine what resources can be requested
#SBATCH --partition=dpart
#SBATCH --gres=gpu:2
#SBATCH --ntasks=8
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 5 python Train.py --name weight-test-acc --graph fully-connected --sgd_steps 1 --personalize 1 --max_sgd 5 --resSize 50 --bs 64 --epoch 5 --description asyncDecentralized --randomSeed 9001 --datasetRoot ./data --outputFolder Output
