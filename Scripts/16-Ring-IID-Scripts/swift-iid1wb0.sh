#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=SWIFT    # sets the job name if not set from environment
#SBATCH --time=04:00:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger   # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger  # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:4
#SBATCH --ntasks=16
#SBATCH --mem 128gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 16 python Train.py --name swift-iid-test1-16W-wb0 --graph ring --sgd_steps 1 --customLR 1 --weight_type swift --momentum 0.9 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --memory_efficient 1 --wb 0 --description SWIFT-No-WB --randomSeed 1236 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name swift-iid-test2-16W-wb0 --graph ring --sgd_steps 1 --customLR 1 --weight_type swift --momentum 0.9 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --memory_efficient 1 --wb 0 --description SWIFT-No-WB --randomSeed 4915 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name swift-iid-test3-16W-wb0 --graph ring --sgd_steps 1 --customLR 1 --weight_type swift --momentum 0.9 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --memory_efficient 1 --wb 0 --description SWIFT-No-WB --randomSeed 3874 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name swift-iid-test4-16W-wb0 --graph ring --sgd_steps 1 --customLR 1 --weight_type swift --momentum 0.9 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --memory_efficient 1 --wb 0 --description SWIFT-No-WB --randomSeed 1134 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name swift-iid-test5-16W-wb0 --graph ring --sgd_steps 1 --customLR 1 --weight_type swift --momentum 0.9 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200 --memory_efficient 1 --wb 0 --description SWIFT-No-WB --randomSeed 9198 --datasetRoot ./data --outputFolder Output
