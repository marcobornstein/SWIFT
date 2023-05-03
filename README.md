

# Shared WaIt-Free Transmission (SWIFT)

## Running SWIFT

After building dependencies, one can test if SWIFT runs by using the following command:

```
mpirun -np 1 python Train.py --description test
```

We include the scripts used to run our experiments within the Scripts folder. These may have to
be altered in order to run on different systems/clusters. We also include a test script (test-script.sh)
which should run SWIFT in parallel with 4 workers.

## Code Dependencies

Below we list the packages and modules that we used when running SWIFT. We believe that are code can work with newer 
packages versions (including PyTorch), however we have not tested on these newer versions. We are hoping to build a 
Python package installer for easier implementation.

We use Python 3.8.2 with the following packages (and versions):
1. torch (1.10.0+cu111)
2. torchvision (0.11.1+cu111)
3. numpy (1.19.5)
4. networkx (2.6.3)
5. mpi4py (2.6.3)
6. Pillow (8.4.0)

Furthermore, for installing mpi4py, we require OpenMPI as a backend. Finally, we use CUDA 11.1.1. Installing PyTorch with CUDA 11.1.1 can be done using
the following command:

```
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Citation
If using SWIFT or this database in further research, please cite either of the following:

BibTeX:
```
@inproceedings{bornstein2023swift,
  title={SWIFT: Rapid Decentralized Federated Learning via Wait-Free Model Communication},
  author={Bornstein, Marco and Rabbani, Tahseen and Wang, Evan and Bedi, Amrit Singh and Huang, Furong},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
MLA:
```
Bornstein, Marco and Rabbani, Tahseen and Wang, Evan and Bedi, Amrit Singh and Huang, Furong. 
"SWIFT: Rapid Decentralized Federated Learning via Wait-Free Model Communication." 
International Conference on Learning Representations. 2023.
```
