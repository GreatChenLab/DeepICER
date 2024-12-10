# DeepICER

## Introduction
This repository contains the PyTorch implementation of **DeepICER** framework. A deep learning-based model for identifying ingredient with cellular response

Code by Fanbo Meng at Chengdu University of Traditional Chinese Medicine.

## System Requirements
The source code developed in Python 3.9 using PyTorch 1.13.1. The required python dependencies are given below. DeepICER is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.

```
pyarrow==11.0.0
prettytable==3.9.0
pytorch==1.13.1
torchvision==0.14.1
torchaudio==0.13.1
pytorch-cuda=11.7
dglteam::dgl-cuda11.7
dgllife==0.3.2
rdkit==2023.9.4
yacs==0.1.8
```
## Inst allation Guide
Clone this Github repo and set up a new conda environment.

```
# create a new conda environment
$ conda create --name deepicer python=3.9
$ conda activate deepicer

# install requried python dependencies
$ conda install pyarrow
$ pip install prettytable dgllife rdkit yacs

# for GPU
$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
$ conda install dglteam::dgl-cuda11.7

# for CPU
$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
$ conda install dglteam::dgl==0.9.1post1


# clone the source code of DeepICER
$ git clone https://github.com/mengfb90/DeepICER.git
$ cd DeepICER
```

## Datasets
The `data` folder contains all experimental data used in DeepICER.

## Run DeepICER on Our Data to Reproduce Results

To train DeepICER, where we provide the basic configurations for all hyperparameters in `config.py`, and customized configurations can be found in `DeepICER.yaml` files.

```
$ python main.py --config DrugBAN.yaml --data data --gpu 0
```

## Infer profile

```
python prediction.py -i infile -o outdir
```
