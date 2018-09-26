<span style="color: green"> Some green text </span>

# 1. BraTs

### 1-1) Overview
<div align="center">
  <img src="https://i.imgur.com/emAFrL1.gif">  <img src="https://i.imgur.com/dGrmh2x.gif">
  <br>
  <em align="center">Fig 1: Brain Complete Tumor Segmention</em>
  <br>
  <img src="https://i.imgur.com/n0WAMwh.gif">  <img src="https://i.imgur.com/PFTwmVb.gif">
  <br>
  <em align="center">Fig 2: Brain Core Tumor Segmention</em>
  <br>
  <br>
  <em align="center"> cf) Ground Truth (Blue) Model Segmentation (Red)</em>
</div>


### 1-2) About
This project is a segmentation model (U-Net) to diagnose brain tumor (Complete, Core, Enhancing). 

### 1-3) U-Net

```bash
models/unet.py
```

![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)


# 2. Dataset
```bash
data_preprocess.py
dataset.py
```
### 2-1) Overview

### 2-2) Data Argumentation

# 3. Train
```bash
train.py
utils.py
```

### 3-1) Loss Function
Dice Coefficient Loss

### 3-2) Optimizer
Adam 

### 3-3) Hyperparameter

# 4. Test
```bash
test.py
```
### 4-1) Test

### 4-2) Checkpoint


# 5. Result
### 5-1) 3D Images
### 5-2) 2D Images
### 5-3) Statistical Index
