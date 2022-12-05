#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=SWIFT     # sets the job name if not set from environment
#SBATCH --time=04:00:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger  # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:2
#SBATCH --ntasks=8
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 8 python Train.py --name swift-iid-test1-8W-no_mem --graph ring --customLR 1 --sgd_steps 1 --weight_type swift --momentum 0.9 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --wb 1 --description SWIFT --randomSeed 3782 --datasetRoot ./data --outputFolder Output
mpirun -np 8 python Train.py --name swift-iid-test2-8W-no_mem --graph ring --customLR 1 --sgd_steps 1 --weight_type swift --momentum 0.9 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --wb 1 --description SWIFT --randomSeed 24 --datasetRoot ./data --outputFolder Output
mpirun -np 8 python Train.py --name swift-iid-test3-8W-no_mem --graph ring --customLR 1 --sgd_steps 1 --weight_type swift --momentum 0.9 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --wb 1 --description SWIFT --randomSeed 332 --datasetRoot ./data --outputFolder Output
mpirun -np 8 python Train.py --name swift-iid-test4-8W-no_mem --graph ring --customLR 1 --sgd_steps 1 --weight_type swift --momentum 0.9 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --wb 1 --description SWIFT --randomSeed 1221 --datasetRoot ./data --outputFolder Output
mpirun -np 8 python Train.py --name swift-iid-test5-8W-no_mem --graph ring --customLR 1 --sgd_steps 1 --weight_type swift --momentum 0.9 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --wb 1 --description SWIFT --randomSeed 1331 --datasetRoot ./data --outputFolder Output
