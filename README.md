# MltGNN: A Multilevel Temporal Graph Neural Network for Workload Prediction in Cloud

## Introduction 
This is a Pytorch implementation of Multilevel Temporal Graph Neural Network for the task of workload prediction of a complex system. This model can extract the time-series information from different dimensional node features. And multilevel graph embedding empower the feature extraction capability of model. To help readers understand the process and reproduce results, we introduce the training environment and training steps in this manuscript.

## Requirements
* python==3.8
* numpy==1.19.5
* pandas==1.2.
* Pytorch==>1.10.1
* argparse

## Dataset overview
We collected the dataset from our own microservices system. It is a shopping microservice system which has 8 nodes, including User, Front end, Order, Payment, Shipping,Catakogue, Cart, and Queue. We initial the edge based on their operation process, that is, there will be an edge if two nodes exchange the message directly. We collect the workload of CUP and Memory of each node. The timestep for data collection is one minute. The sample of each node will be a matrix consisting of 50 timesteps in same time to retain the time series information, and the label for training is the value in 1/3/5/10/20 timesteps later.

## Run the demo

```bash
python action_one.py
python ML_benchmark.py 
```

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), 
* an N by D feature matrix (D is the number of features per node), and
* an N by E binary label matrix (E is the number of classes).

Have a look at the `load_data()` function in `utils.py` for an example.

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://www.cs.umd.edu/~sen/lbc-proj/LBC.html. In our version (see `data` folder) we use dataset splits provided by https://github.com/kimiyoung/planetoid (Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861), ICML 2016). 

You can specify a dataset as follows:

```bash
python train.py --dataset citeseer
```

(or by editing `train.py`)

## Models

You can choose between the following models: 
* `gcn`: Graph convolutional network (Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907), 2016)
* `gcn_cheby`: Chebyshev polynomial version of graph convolutional network as described in (Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), NIPS 2016)
* `dense`: Basic multi-layer perceptron that supports sparse inputs

## Graph classification

Our framework also supports batch-wise classification of multiple graph instances (of potentially different size) with an adjacency matrix each. It is best to concatenate respective feature matrices and build a (sparse) block-diagonal matrix where each block corresponds to the adjacency matrix of one graph instance. For pooling (in case of graph-level outputs as opposed to node-level outputs) it is best to specify a simple pooling matrix that collects features from their respective graph instances, as illustrated below:

![graph_classification](https://user-images.githubusercontent.com/7347296/34198790-eb5bec96-e56b-11e7-90d5-157800e042de.png)


## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```
