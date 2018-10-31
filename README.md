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

 
<br>
<br>

### 1-2) About
  **This project is a segmentation model to diagnose brain tumor (Complete, Core) using BraTS 2016, 2017 dataset.**
  >[BraTS](http://www.med.upenn.edu/sbia/brats2018.html) has always been focusing on the evaluation of state-of-the-art methods for the
  segmentation of brain tumors in multimodal magnetic resonance imaging (MRI) scans. BraTS 2018 utilizes multi-institutional pre-
  operative MRI scans and focuses on the segmentation of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, 
  namely gliomas. Furthemore, to pinpoint the clinical relevance of this segmentation task, BraTS’18 also focuses on the prediction of 
  patient overall survival, via integrative analyses of radiomic features and machine learning algorithms.


<br>
<br>

### 1-3) Models

  - **U-Net**

```bash
pytorch/models/unet.py
```

<div align="center">
  <img src="https://i.imgur.com/OXtVFvT.png">
  <br>
  <br>
  <em align="center">Fig 3: U-Net Diagram </em>
  <br>
</div>

<br>
<br>

  - **PSPNet**

```bash
pytorch/models/pspnet.py
```

<div align="center">
  <img src="https://i.imgur.com/y8M2IzT.png">
  <br>
  <br>
  <em align="center">Fig 4: PSPNet Diagram </em>
  <br>
</div>

<br>
<br>

 - **DeepLab V3 +**

```bash
pytorch/models/deeplab.py
```

<div align="center">
  <img src="https://i.imgur.com/5IBKzDx.png">
  <br>
  <br>
  <em align="center">Fig 5: DeepLab V3+ Diagram </em>
  <br>
</div>

<br>
<br>


# 2. Dataset


### 2-1) Overview

**Multimodal MRI Dataset**

*File:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;A File has Multi-Modal MRI Data of one person*

*File Format:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
nii.gz*

*Image Shape:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
240(Slide Width) × 240(Slide Height) × 155(Number of Slide) × 4(Multi-mode)*

*Image Mode:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
4 (Multi-mode)*

<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
channel 0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
channel 1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
channel 2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
channel 3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

<div align="center">
  <img src="https://i.imgur.com/xXkKu2L.png">
</div>

<br>

*Label Shape:&nbsp;&nbsp;&nbsp;
channel 0: background*

*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;channel 1: necrotic and non-enhancing tumor*

*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;channel 2: edema*

*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;channel 3: enhancing tumor*

<br>


### 2-2) Data Argumentation

![](https://i.imgur.com/yQmxilp.png)

![](https://i.imgur.com/6Ovl6Xd.png)

### 2-3) Code

```bash
pytorch/preprocess.py
pytorch/dataset.py
```

<br>

`preprocess.py`: Code for data pre-processing. Using this, original image(240×240×155×4) can be diveded into 155 image pieces(240×240) of the specific mode. Also, original label(240×240×155) can be divided into 155 label pieces.

`dataset.py`: Code for Prepareing dataset and dataloader for [Pytorch](https://pytorch.org/docs/stable/index.html) modules

<br>
<br>

# 3. Train

### 3-1) Loss Function
  [Dice Coefficient Loss](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)

![](https://i.imgur.com/aGUbIeU.png)

### 3-2) Optimizer
  [Adam Optimizer](https://arxiv.org/pdf/1412.6980.pdf)

  [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
  <br>

### 3-3) Hyperparameter
  learning rate = 1e-4

  maximum number of epochs = 100

  Weights Init: Normal Distribution (mean:0, std:0.01)
  
  Bias Init: Initialized as 0

<br>

### 3-4) Code

```bash
pytorch/train.py
pytorch/utils.py
```

`train.py`: Code for training model and getting several inputs

`utils.py`: Code for loss Function, utils Functions, UI Functions, and etc

<br>
<br>

# 4. Test

```bash
pytorch/test.py
```

`test.py`: Code for testing MRI inputs

<br>

# 5. Result

### 5-1) Prediction

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

<br>

### 5-2) Statistical Indices

<div align="center">
  <img src="https://i.imgur.com/tI9PXLR.png">  
  <br>
  <em align="center">Fig 3: Statistical Indices</em>
  <br>
</div>


<br>
<br>

# 6. Reference

[1] [Automatic Brain Tumor Detection and Segmentation Using U-Net Based Fully Convolutional Networks arugmentation](https://arxiv.org/pdf/1705.03820.pdf)

[2] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

[3] [Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf)

[4] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)

