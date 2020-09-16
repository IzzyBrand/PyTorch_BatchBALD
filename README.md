# PyTorch_BatchBALD
A PyTorch implementation of BatchBALD on the MNIST dataset.

Bayesian Active Learning by Disagreement (BALD) is a mutual-information-based method for active learning -- selecting the most informative samples from a pool of data.

 * Followed description of BALD in [Deep Bayesian Active Learning with Image Data](https://arxiv.org/pdf/1703.02910.pdf)
 * Follow description of BatchBALD in [BatchBALD: Efficient and Diverse Batch Acquisition
for Deep Bayesian Active Learning](https://arxiv.org/pdf/1906.08158.pdf)

### Installation

Requires Python3.

```
pip3 install torch torchvision matplotlib numpy
```

### Usage

```
python3 main.py
```
