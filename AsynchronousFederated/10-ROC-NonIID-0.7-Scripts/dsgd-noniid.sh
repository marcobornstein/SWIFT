#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=DSGD-0.7     # sets the job name if not set from environment
#SBATCH --time=04:00:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger  # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:3
#SBATCH --ntasks=10
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 10 python Train.py  --graph clique-ring --num_clusters 3 --name dsgd-noniid-0.7-test1-10W --comm_style d-sgd --momentum 0.9 --lr 0.8 --degree_noniid 0.7 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description DSGD-paper --randomSeed 1000 --datasetRoot ./data --outputFolder Output
mpirun -np 10 python Train.py  --graph clique-ring --num_clusters 3 --name dsgd-noniid-0.7-test2-10W --comm_style d-sgd --momentum 0.9 --lr 0.8 --degree_noniid 0.7 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description DSGD-paper --randomSeed 2500 --datasetRoot ./data --outputFolder Output
mpirun -np 10 python Train.py  --graph clique-ring --num_clusters 3 --name dsgd-noniid-0.7-test3-10W --comm_style d-sgd --momentum 0.9 --lr 0.8 --degree_noniid 0.7 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description DSGD-paper --randomSeed 250 --datasetRoot ./data --outputFolder Output
mpirun -np 10 python Train.py  --graph clique-ring --num_clusters 3 --name dsgd-noniid-0.7-test4-10W --comm_style d-sgd --momentum 0.9 --lr 0.8 --degree_noniid 0.7 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description DSGD-paper --randomSeed 225 --datasetRoot ./data --outputFolder Output
mpirun -np 10 python Train.py  --graph clique-ring --num_clusters 3 --name dsgd-noniid-0.7-test5-10W --comm_style d-sgd --momentum 0.9 --lr 0.8 --degree_noniid 0.7 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description DSGD-paper --randomSeed 200 --datasetRoot ./data --outputFolder Output
