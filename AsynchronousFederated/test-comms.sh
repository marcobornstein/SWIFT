#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=AsyncDecentralized     # sets the job name if not set from environment
#SBATCH --time=00:55:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=furongh    # set QOS, this will determine what resources can be requested
#SBATCH --qos=medium    # set QOS, this will determine what resources can be requested
#SBATCH --partition=dpart
#SBATCH --gres=gpu:2
#SBATCH --ntasks=12
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 11 python Train.py  --graph clique-ring --num_clusters 3 --name dsgd-test --comm_style d-sgd --resSize 50 --bs 64 --epoch 50 --description asyncDecentralized --randomSeed 9001 --datasetRoot ./data --outputFolder Output
# mpirun -np 5 python Train.py  --graph fully-connected --name pdsgd-test --comm_style pd-sgd --i1 1 --resSize 50 --bs 64 --epoch 25  --description asyncDecentralized --randomSeed 9001 --datasetRoot ./data --outputFolder Output
# mpirun -np 5 python Train.py  --graph fully-connected --name ldsgd-test  --comm_style ld-sgd --i1 1 --i2 2 --resSize 50 --bs 64 --epoch 25  --description asyncDecentralized --randomSeed 9001 --datasetRoot ./data --outputFolder Output
# mpirun -np 6 python Train.py  --graph erdos-renyi --name padfed-test --sgd_steps 2 --num_clusters 1 --personalize 1 --max_sgd 10 --resSize 50 --bs 64 --epoch 100  --description asyncDecentralized --randomSeed 9001 --datasetRoot ./data --outputFolder Output
