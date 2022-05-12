#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=SWIFT     # sets the job name if not set from environment
#SBATCH --time=4:00:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=furongh    # set QOS, this will determine what resources can be requested
#SBATCH --qos=high  # set QOS, this will determine what resources can be requested
#SBATCH --gres=gpu:4
#SBATCH --ntasks=16
#SBATCH --mem 128gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 16 python Train.py --name swift-iid-test1-16W-no_mem --graph ring --sgd_steps 1 --weight_type swift --momentum 0.9 --customLR 1 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --wb 1 --description SWIFT --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name swift-iid-test2-16W-no_mem --graph ring --sgd_steps 1 --weight_type swift --momentum 0.9 --customLR 1 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --wb 1 --description SWIFT --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name swift-iid-test3-16W-no_mem --graph ring --sgd_steps 1 --weight_type swift --momentum 0.9 --customLR 1 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --wb 1 --description SWIFT --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name swift-iid-test4-16W-no_mem --graph ring --sgd_steps 1 --weight_type swift --momentum 0.9 --customLR 1 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --wb 1 --description SWIFT --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 16 python Train.py --name swift-iid-test5-16W-no_mem --graph ring --sgd_steps 1 --weight_type swift --momentum 0.9 --customLR 1 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 300 --wb 1 --description SWIFT --randomSeed 9001 --datasetRoot ./data --outputFolder Output
