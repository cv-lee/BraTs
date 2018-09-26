# 1. BraTs   (Brain Tumor Segmentation)

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
  <img src="https://placehold.it/15/1589F0/000000?text=+">
  <em align="center">Ground Truth</em>
  <br>
  <img src="https://placehold.it/15/f03c15/000000?text=+">
  <em align="center">Prediction</em>
  <br>
</div>


### 1-2) About
**This project is a segmentation model to diagnose brain tumor (Complete, Core) using BraTS 2016, 2017 dataset.**
>[BraTS](http://www.med.upenn.edu/sbia/brats2018.html) has always been focusing on the evaluation of state-of-the-art methods for the segmentation of brain tumors in multimodal magnetic resonance imaging (MRI) scans. BraTS 2018 utilizes multi-institutional pre-operative MRI scans and focuses on the segmentation of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas. Furthemore, to pinpoint the clinical relevance of this segmentation task, BraTS’18 also focuses on the prediction of patient overall survival, via integrative analyses of radiomic features and machine learning algorithms.


### 1-3) U-Net

```bash
models/unet.py
```

![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)


# 2. Dataset

*Multimodal MRI Dataset*

*File:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
A File has Multi-Modal MRI Data of one person*

*File Format:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
nii.gz*

*Data Shape:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
240(Slide Width) × 240(Slide Height) × 155(Number of Slide) × 4(Multi-mode)*

*Data Mode:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
4 (Multi-mode)*

*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
channel 0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
channel 1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
channel 2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
channel 3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*

<div align="center">
  <img src="https://i.imgur.com/xXkKu2L.png">
</div>




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
