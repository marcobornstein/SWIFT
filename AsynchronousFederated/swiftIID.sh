#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=AdFed     # sets the job name if not set from environment
#SBATCH --time=06:45:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger  # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:4
#SBATCH --ntasks=16
#SBATCH --mem 128gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 16 python Train.py --name adfed-iid-test1-10W --graph ring --num_clusters 3 --sgd_steps 2 --personalize 0 --max_sgd 5 --resSize 18 --bs 32 --epoch 300 --description AdFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name adfed-iid-test2-10W --graph ring --num_clusters 3 --sgd_steps 2 --personalize 0 --max_sgd 5 --resSize 18 --bs 32 --epoch 300 --description AdFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name adfed-iid-test3-10W --graph ring --num_clusters 3 --sgd_steps 2 --personalize 0 --max_sgd 5 --resSize 18 --bs 32 --epoch 300 --description AdFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name adfed-iid-test4-10W --graph ring --num_clusters 3 --sgd_steps 2 --personalize 0 --max_sgd 5 --resSize 18 --bs 32 --epoch 300 --description AdFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name adfed-iid-test5-10W --graph ring --num_clusters 3 --sgd_steps 2 --personalize 0 --max_sgd 5 --resSize 18 --bs 32 --epoch 300 --description AdFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
