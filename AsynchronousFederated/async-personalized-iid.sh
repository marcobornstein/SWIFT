#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=PadFed     # sets the job name if not set from environment
#SBATCH --time=04:45:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=furongh    # set QOS, this will determine what resources can be requested
#SBATCH --qos=medium    # set QOS, this will determine what resources can be requested
#SBATCH --partition=dpart
#SBATCH --gres=gpu:2
#SBATCH --ntasks=8
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 6 python Train.py --name async-pers-iid-5sgdmax-2sgdstep-fc-5-test1 --graph fully-connected --sgd_steps 2 --personalize 1 --max_sgd 5 --noniid 0 --resSize 50 --bs 64 --epoch 200 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 6 python Train.py --name async-pers-iid-5sgdmax-2sgdstep-fc-5-test2 --graph fully-connected --sgd_steps 2 --personalize 1 --max_sgd 5 --noniid 0 --resSize 50 --bs 64 --epoch 200 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 6 python Train.py --name async-pers-iid-5sgdmax-2sgdstep-fc-5-test3 --graph fully-connected --sgd_steps 2 --personalize 1 --max_sgd 5 --noniid 0 --resSize 50 --bs 64 --epoch 200 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 6 python Train.py --name async-pers-iid-5sgdmax-2sgdstep-fc-5-test4 --graph fully-connected --sgd_steps 2 --personalize 1 --max_sgd 5 --noniid 0 --resSize 50 --bs 64 --epoch 200 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
mpirun -np 6 python Train.py --name async-pers-iid-5sgdmax-2sgdstep-fc-5-test5 --graph fully-connected --sgd_steps 2 --personalize 1 --max_sgd 5 --noniid 0 --resSize 50 --bs 64 --epoch 200 --description PadFed --randomSeed 9001 --datasetRoot ./data --outputFolder Output
