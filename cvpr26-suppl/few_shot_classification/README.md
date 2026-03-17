## Requirements
### Installation
Create a conda environment and install dependencies:
```
conda create -n reevaluate-fewshot python=3.9
conda activate reevaluate-fewshot

pip install -r requirements.txt
```

### Dataset
Follow [DATASET.md](../DATASET.md) to install datasets.


## Pre-extract features
```bash
python preextract_features.py
```
The following parameters are available:
```bash
--config configs/few_shots/stanford_cars  # choose from the available files in configs/few_shots
--backbone RN50  # choices=['RN50', 'ViT-B/16', 'ViT-B/32'],
```
It will store the extracted embeddings in `data/image_features` and load them automatically on inference below.
## Run Few-shot Classification
```bash
python main_few_shots.py 
```

The following parameters are available:
```bash
--dataset stanford_cars  # choices=['stanford_cars', 'oxford_pets', 'oxford_flowers', 'fgvc_aircraft', 'dtd', 'eurosat', 'food-101', 'caltech101', 'ucf101', 'sun397', 'imagenet'
--backbone RN50  # choices=['RN50', 'ViT-B/16', 'ViT-B/32', 'ViT-L/14', 'siglip-ViT-B-16', 'siglip2-ViT-B-16', 'dinov2_vitb14','dinov3_vitl16']
--method project  # choices=['project', 'prototype', 'gda'] (project is ours, gda=lda)
```
You should then get the accuracy results printed.

Note: The code is adapted from the ICLR 2024 LDA baseline.