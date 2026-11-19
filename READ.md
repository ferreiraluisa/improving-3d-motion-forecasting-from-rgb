# Improving 3D Human Motion Forecasting from RGB Videos using Auxiliary Tasks

This repository contains the method and experiments of my Master's Thesis at Univeristy of Bonn.

The thesis is related to improving 3D Human forescasting from an autoregressive approach by exploring different feature extractors and modernizing the archicteture, from stacked 1d convolutions to transformer-based architectures. 

## Getting Started

In this thesis, we evaluate feature extractors from three pre-trained models for Human Mesh Recovery (HMR). To reproduce our experiments, first download the pre-trained model weights and extract them into the corresponding directories.

Create the required directories:

```bash
mkdir -p models/{hmr,hmr2,wham}
```

### 1. HMR

Repository: [HMR](https://github.com/akanazawa/hmr)

```bash
wget -O models/hmr/models.tar.gz \
    https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz

tar -xf models/hmr/models.tar.gz -C models/hmr
```

### 2. HMR 2.0

Repository: [4D-Humans](https://github.com/shubham-goel/4D-Humans)

```bash
wget -O models/hmr2/hmr2_data.tar.gz \
    https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz

tar -xf models/hmr2/hmr2_data.tar.gz -C models/hmr2
```

### 3. WHAM

Website: [WHAM](https://wham.is.tue.mpg.de/)

Download the pre-trained weights from the [WHAM project website](https://wham.is.tue.mpg.de/) or the following Google Drive link:

```bash
wget -O models/wham/wham_data.zip \
    'https://drive.google.com/uc?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB&export=download&confirm=t'
```

Extract the downloaded archive according to its format. For a ZIP archive, run:

```bash
unzip models/wham/wham_data.zip -d models/wham
```

**Note:** The Google Drive download may require confirmation or additional steps. Verify that the downloaded file is a valid archive before extracting it.

After downloading and extracting the weights, the directory structure should contain the following folders:

```text
models/
├── hmr/
├── hmr2/
└── wham/
```

## Datasets
We use two datasets: [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/).

For Human3.6M, the **3D and 2D ground truth** was extracted with the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/main/data/prepare_data_h36m.py) preprocessing script:
```bash
python prepare_data_h36m.py --from-source-cdf /home/group-cvg/datasets/human3.6
```
This produces:
- `data_3d_h36m.npz` — 3D ground-truth joint positions in meters (world space)
- `data_2d_h36m_gt.npz` — 2D projections in pixel space for all 4 cameras

We use the 2D data to crop each frame around the subject.

## Preprocessing
The input to our model is RGB video, which we encode with pretrained feature extractors. To speed up training, we extract the features once in advance and feed these precomputed features to the model.

Feature extraction is done by `dataset/feats.py` (Human3.6M) and `dataset/feats_3dpw.py` (3DPW). To run it, use `scripts/extract_features.sh`:
```bash
bash scripts/extract_features.sh h36m
bash scripts/extract_features.sh 3dpw
```
We tested four feature extractors, described in the thesis: HMR (ResNet-50), HMR2.0 ViT (mean-pooled), HMR2.0 cross-attention token, and WHAM. 

For WHAM, the 3D joints inside the features are in the COCO-17 format, which does not match our ground truth. We therefore train a small adapter that maps them to the H3.6M skeleton (17 joints) for Human3.6M and to the SMPL skeleton (24 joints) for 3DPW. The same script trains the adapter and then writes a new feature file with the corrected skeleton:
```bash
bash scripts/wham_adapter.sh h36m
bash scripts/wham_adapter.sh 3dpw
```
For 3DPW the features grow from 563 to 584 dimensions (512 context + 24×3 joints).

## Baseline: Predicting Human Dynamics (PHD)
Our adapted implementation of ["Predicting 3D Human Dynamics from Video"](https://arxiv.org/pdf/1908.04781) is in `models/phd_model.py`. Training has two phases: phase 1 trains `f_movie` and `f_3D` to reconstruct the 3D pose, and phase 2 freezes `f_3D` and trains `f_movie` and `f_AR` to forecast future poses. To train and test, use `scripts/train_test_phd.sh`:
```bash
bash scripts/train_test_phd.sh h36m train   # phase 1, then phase 2
bash scripts/train_test_phd.sh h36m test
```

## Proposed Method: MotionST
We propose to improve PHD with a transformer-based model and more recent feature extractors than the ResNet used in the original PHD. The model is in `models/motionst.py`. Unlike PHD, it is trained in a single stage: it reconstructs the observed frames and predicts the future frames at the same time. To train and test, use `scripts/train_test_motionst.sh`:
```bash
bash scripts/train_test_motionst.sh h36m train
bash scripts/train_test_motionst.sh h36m test
```

## Case Study: Auxiliary Tasks
We also study whether auxiliary tasks improve the model. During training only, the WHAM input is corrupted in one of three ways, and the model must still recover the ground-truth pose:
- `joint`: random joints are masked
- `frame`: all joints of random past frames are masked
- `noise`: Gaussian noise is added to the WHAM joints

Auxiliary tasks require WHAM features. To train and test, use `scripts/auxiliary_task.sh`:
```bash
bash scripts/auxiliary_task.sh h36m frame
```

## Alternative Architecture: MotionSTAR


## Author

| [Luísa Ferreira](https://ferreiraluisa.github.io/luisaferreira/) |
| :------------------------------------------------------------: | 
| Msc. Student¹                                                 | 
| <s26ldeso@uni-bonn.de>                                       

¹University of Bonn \
Bonn, North Rhine-Westphalia, Germany

## Acknowledgements

I would like to thank Fatemeh Jabbari and Marius Bock for their guidance and support as advisors of this thesis. Claude Opus 5.5 (Anthropic) helped write the bash scripts and edit the documentation in this repository.

---

### Enjoy it! :smiley:


