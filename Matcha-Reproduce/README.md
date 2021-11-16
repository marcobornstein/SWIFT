# MATCHA Reproducing and Cleaning Code

In this folder, we fix the code from the MATCHA: Communication-Efficient Decentralized SGD code repository (CODE: https://github.com/JYWa/MATCHA, PAPER: https://arxiv.org/abs/1905.09435). Furthermore, we reproduce their results training Resnet to classify Cifar10 using a Decentralized Communicator. Major overhauls were necessary to accomplish this reproduction. Below we provide details on what dependencies are needed to use our code.

## Dependencies

Our code was run on the University of Maryland Center for Machine Learning (CML) cluster (https://wiki.umiacs.umd.edu/umiacs/index.php/CML). Within the CML cluster, Python 3.8.2 is used. In our virtual environment, we use Pytorch 1.10.0 and torchvision 0.11.1. The decentralized communication is accomplished using mpi4py (3.1.2). Other packages necessary to run the code include cvxpy (1.1.17) and networkx (2.6.3)

