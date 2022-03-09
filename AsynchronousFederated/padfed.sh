#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=PadFed     # sets the job name if not set from environment
#SBATCH --time=04:45:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=furongh    # set QOS, this will determine what resources can be requested
#SBATCH --qos=high    # set QOS, this will determine what resources can be requested
#SBATCH --partition=dpart
#SBATCH --gres=gpu:3
#SBATCH --ntasks=12
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 11 python Train.py --name padfed-noniid-test1-10W --graph clique-ring --num_clusters 3 --sgd_steps 2 --personalize 1 --max_sgd 5 --degree_noniid 1 --noniid 1 --resSize 50 --bs 64 --epoch 200 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 11 python Train.py --name padfed-noniid-test2-10W --graph clique-ring --num_clusters 3 --sgd_steps 2 --personalize 1 --max_sgd 5 --degree_noniid 1 --noniid 1 --resSize 50 --bs 64 --epoch 200 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 11 python Train.py --name padfed-noniid-test3-10W --graph clique-ring --num_clusters 3 --sgd_steps 2 --personalize 1 --max_sgd 5 --degree_noniid 1 --noniid 1 --resSize 50 --bs 64 --epoch 200 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 11 python Train.py --name padfed-noniid-test4-10W --graph clique-ring --num_clusters 3 --sgd_steps 2 --personalize 1 --max_sgd 5 --degree_noniid 1 --noniid 1 --resSize 50 --bs 64 --epoch 200 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 11 python Train.py --name padfed-noniid-test5-10W --graph clique-ring --num_clusters 3 --sgd_steps 2 --personalize 1 --max_sgd 5 --degree_noniid 1 --noniid 1 --resSize 50 --bs 64 --epoch 200 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
