#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=LDSGD     # sets the job name if not set from environment
#SBATCH --time=10:30:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=furongh    # set QOS, this will determine what resources can be requested
#SBATCH --qos=high    # set QOS, this will determine what resources can be requested
#SBATCH --gres=gpu:4
#SBATCH --ntasks=16
#SBATCH --mem 128gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 16 python Train.py  --graph ring --num_clusters 3 --name ldsgd-iid-test1-16W --comm_style ld-sgd --momentum 0.9 --customLR 1 --i1 1 --i2 2 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --description LDSGD --randomSeed 143 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py  --graph ring --num_clusters 3 --name ldsgd-iid-test2-16W --comm_style ld-sgd --momentum 0.9 --customLR 1 --i1 1 --i2 2 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --description LDSGD --randomSeed 1943 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py  --graph ring --num_clusters 3 --name ldsgd-iid-test3-16W --comm_style ld-sgd --momentum 0.9 --customLR 1 --i1 1 --i2 2 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --description LDSGD --randomSeed 1832 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py  --graph ring --num_clusters 3 --name ldsgd-iid-test4-16W --comm_style ld-sgd --momentum 0.9 --customLR 1 --i1 1 --i2 2 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --description LDSGD --randomSeed 9234 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py  --graph ring --num_clusters 3 --name ldsgd-iid-test5-16W --comm_style ld-sgd --momentum 0.9 --customLR 1 --i1 1 --i2 2 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --description LDSGD --randomSeed 634 --datasetRoot ./data --outputFolder Output
