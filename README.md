# Causality Reasoning

[![LICENSE](https://img.shields.io/badge/license-GPL-green)](http://www.buaamsc.com:7929/SY2006304/ACR.pytorch/-/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.6.0-%237732a8)](https://pytorch.org/get-started/previous-versions/)

Code for Causality Reasoning.:sunny:

## Recent Updates
- [x] 2021.09.14 Add all codes for experiments.
- [x] 2021.11.15 Modify the model and get new experimental results.
- [ ] TODO: Add link to our paper.

## Contents
1. [Overview](#Overview)
2. [Prepare the Dataset](#Prepare-the-Dataset)
3. [Get Checkpoints](#Get-Checkpoints)
4. [Get Start](#Get-Checkpoints)
5. [Citations](#Citations)

## Overview
### Files
* visualization<br/>
——visualize.ipynb Object detection and prediction result distribution visualization.
* atomic_pretrain.py Used for pretraining on ATOMIC.
* atomic_reader.py Data processing script for ATOMIC.
* main.py Train and Test for our model.
* requirements.txt Required runtime environment.

### Experimental results on Vis-Causal dataset, this codebase achieves new state-of-the-art
Models | Recall@1 | Recall@5 | Recall@10
-- | -- | -- | -- 
Random Guess | 2.13 | 15.25 | 30.14
VCC | 8.87 | 34.15 | 63.12
iReason | 9.21 | 35.87 | 63.51
ours w/o fusion | 17.02 | 49.29 | 70.21
ours | __17.38__ | __50.00__ | __71.28__

More results are shown in our paper.

### Pay Attention
+ The images of the *validation* and *test* sets in the original dataset are interleaved, while we have fixed this problem.
+ We use the Faster R-CNN model from [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), objects with a confidence greater than 0.3 are selected (predicted label, bounding box and RoI feature) to be the inputs.
+ The experiments are conducted on a singal NVIDIA Tesla V100 GPU.

## Prepare the Dataset
Dataset is available at [here](https://bhpan.buaa.edu.cn:443/link/42E08E6F849ED6A228776AA34A6E2C33).<br/>
Including ATOMIC, Vis-Causal(original data and detected objects) and GloVe.

## Get Checkpoints
Checkpoints are available at [here](https://bhpan.buaa.edu.cn:443/link/F81D6A3E3EE44C71CEC0B1B0B8DE52E4).<br/>
Including pre-trained models and the final checkpoints.

## Get Start
Use this command to start training the model:
```python
nohup python -u main.py >main.log &
```
To test the model, remove the commented part in [main.py](main.py).

## Citations
TBD.
