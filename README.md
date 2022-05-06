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
```
