#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=DSGD     # sets the job name if not set from environment
#SBATCH --time=10:30:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:4
#SBATCH --ntasks=10
#SBATCH --mem 128gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

# module load openmpi
# module load cuda/11.1.1

mpirun -np 10 python Train.py  --graph ring --num_clusters 3 --name dsgd-noniid-test1-10W --comm_style d-sgd --degree_noniid 1 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description DSGD-paper --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 10 python Train.py  --graph ring --num_clusters 3 --name dsgd-noniid-test2-10W --comm_style d-sgd --degree_noniid 1 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description DSGD-paper --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 10 python Train.py  --graph ring --num_clusters 3 --name dsgd-noniid-test3-10W --comm_style d-sgd --degree_noniid 1 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description DSGD-paper --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 10 python Train.py  --graph ring --num_clusters 3 --name dsgd-noniid-test4-10W --comm_style d-sgd --degree_noniid 1 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description DSGD-paper --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 10 python Train.py  --graph ring --num_clusters 3 --name dsgd-noniid-test5-10W --comm_style d-sgd --degree_noniid 1 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description DSGD-paper --randomSeed 9001 --datasetRoot ./data --outputFolder Output
