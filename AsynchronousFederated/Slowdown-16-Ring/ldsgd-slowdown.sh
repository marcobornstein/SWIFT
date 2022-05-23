#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=LDSGD-slow     # sets the job name if not set from environment
#SBATCH --time=06:30:00     # how long you think your job will take to complete; format=hh:mm:ss
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

mpirun -np 16 python Train_Slowdown.py  --graph ring --name ldsgd-slowdown2-test1-16W --comm_style ld-sgd --i1 1 --i2 2 --momentum 0.9 --slowdown 2 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 100 --description LDSGD-paper --randomSeed 2332 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train_Slowdown.py  --graph ring --name ldsgd-slowdown2-test2-16W --comm_style ld-sgd --i1 1 --i2 2 --momentum 0.9 --slowdown 2 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 100 --description LDSGD-paper --randomSeed 2442 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train_Slowdown.py  --graph ring --name ldsgd-slowdown2-test3-16W --comm_style ld-sgd --i1 1 --i2 2 --momentum 0.9 --slowdown 2 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 100 --description LDSGD-paper --randomSeed 2992 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train_Slowdown.py  --graph ring --name ldsgd-slowdown2-test4-16W --comm_style ld-sgd --i1 1 --i2 2 --momentum 0.9 --slowdown 2 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 100 --description LDSGD-paper --randomSeed 4844 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train_Slowdown.py  --graph ring --name ldsgd-slowdown2-test5-16W --comm_style ld-sgd --i1 1 --i2 2 --momentum 0.9 --slowdown 2 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 100 --description LDSGD-paper --randomSeed 2900 --datasetRoot ./data --outputFolder Output
