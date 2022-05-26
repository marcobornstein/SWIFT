#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=SWIFT-0.25     # sets the job name if not set from environment
#SBATCH --time=05:30:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger   # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger  # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:3
#SBATCH --ntasks=10
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 10 python Train.py --name swift-noniid-0.25-test1-10W-no_mem --graph clique-ring --num_clusters 3 --sgd_steps 1 --weight_type swift --momentum 0.9 --lr 0.8 --degree_noniid 0.25 --noniid 1 --resSize 18 --bs 32 --epoch 300 --wb 1 --description SWIFT --randomSeed 1333 --datasetRoot ./data --outputFolder Output
mpirun -np 10 python Train.py --name swift-noniid-0.25-test2-10W-no_mem --graph clique-ring --num_clusters 3 --sgd_steps 1 --weight_type swift --momentum 0.9 --lr 0.8 --degree_noniid 0.25 --noniid 1 --resSize 18 --bs 32 --epoch 300 --wb 1 --description SWIFT --randomSeed 1346 --datasetRoot ./data --outputFolder Output
mpirun -np 10 python Train.py --name swift-noniid-0.25-test3-10W-no_mem --graph clique-ring --num_clusters 3 --sgd_steps 1 --weight_type swift --momentum 0.9 --lr 0.8 --degree_noniid 0.25 --noniid 1 --resSize 18 --bs 32 --epoch 300 --wb 1 --description SWIFT --randomSeed 1345 --datasetRoot ./data --outputFolder Output
mpirun -np 10 python Train.py --name swift-noniid-0.25-test4-10W-no_mem --graph clique-ring --num_clusters 3 --sgd_steps 1 --weight_type swift --momentum 0.9 --lr 0.8 --degree_noniid 0.25 --noniid 1 --resSize 18 --bs 32 --epoch 300 --wb 1 --description SWIFT --randomSeed 9183 --datasetRoot ./data --outputFolder Output
mpirun -np 10 python Train.py --name swift-noniid-0.25-test5-10W-no_mem --graph clique-ring --num_clusters 3 --sgd_steps 1 --weight_type swift --momentum 0.9 --lr 0.8 --degree_noniid 0.25 --noniid 1 --resSize 18 --bs 32 --epoch 300 --wb 1 --description SWIFT --randomSeed 1337 --datasetRoot ./data --outputFolder Output
