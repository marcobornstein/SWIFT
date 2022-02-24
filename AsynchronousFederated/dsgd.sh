#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=DSGD     # sets the job name if not set from environment
#SBATCH --time=05:30:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:2
#SBATCH --ntasks=8
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 5 python Train.py  --graph fully-connected --name test1-dsgd-iid-fc-4 --comm_style d-sgd --degree_noniid 0.7 --noniid 0 --resSize 50 --bs 64 --epoch 200 --description DSGD --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 5 python Train.py  --graph fully-connected --name test2-dsgd-iid-fc-4 --comm_style d-sgd --degree_noniid 0.7 --noniid 0 --resSize 50 --bs 64 --epoch 200 --description DSGD --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 5 python Train.py  --graph fully-connected --name test3-dsgd-iid-fc-4 --comm_style d-sgd --degree_noniid 0.7 --noniid 0 --resSize 50 --bs 64 --epoch 200 --description DSGD --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 5 python Train.py  --graph fully-connected --name test4-dsgd-iid-fc-4 --comm_style d-sgd --degree_noniid 0.7 --noniid 0 --resSize 50 --bs 64 --epoch 200 --description DSGD --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 5 python Train.py  --graph fully-connected --name test5-dsgd-iid-fc-4 --comm_style d-sgd --degree_noniid 0.7 --noniid 0 --resSize 50 --bs 64 --epoch 200 --description DSGD --randomSeed 9001 --datasetRoot ./data --outputFolder Output
