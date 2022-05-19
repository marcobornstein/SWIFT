#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=PDSGD     # sets the job name if not set from environment
#SBATCH --time=04:00:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger  # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:2
#SBATCH --ntasks=4
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 4 python Train.py  --graph ring --num_clusters 3 --name pdsgd-iid-test1-4W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --customLR 1 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --description PDSGD --randomSeed 2282 --datasetRoot ./data --outputFolder Output
mpirun -np 4 python Train.py  --graph ring --num_clusters 3 --name pdsgd-iid-test2-4W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --customLR 1 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --description PDSGD --randomSeed 9867 --datasetRoot ./data --outputFolder Output
mpirun -np 4 python Train.py  --graph ring --num_clusters 3 --name pdsgd-iid-test3-4W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --customLR 1 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --description PDSGD --randomSeed 6666 --datasetRoot ./data --outputFolder Output
mpirun -np 4 python Train.py  --graph ring --num_clusters 3 --name pdsgd-iid-test4-4W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --customLR 1 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --description PDSGD --randomSeed 7777 --datasetRoot ./data --outputFolder Output
mpirun -np 4 python Train.py  --graph ring --num_clusters 3 --name pdsgd-iid-test5-4W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --customLR 1 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --description PDSGD --randomSeed 8888 --datasetRoot ./data --outputFolder Output
