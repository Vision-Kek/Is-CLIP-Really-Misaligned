## Installation Guide 

Create and Activate Conda Environment
```bash
conda create -n reevaluate-retrieval python=3.10
conda activate reevaluate-retrieval
```

Installing Dassl Library and Requirements
```bash
pip install git+https://github.com/KaiyangZhou/Dassl.pytorch
chmod +x install_requirements.sh
./install_requirements.sh
```

If you want to use SigLIP, SLIP or DINO, download the respective checkpoint and set the directory 
`MODEL_DIR = '/path/to/checkpoint/dir'` in the respective model python file.

Follow [DATASET.md](../DATASET.md) to install datasets.

## Run retrieval
```bash
python src/retrieval.py 
```
The following parameters are available:
```bash
--dataroot /path/to/your/dataset/root/dir
--dataset_name stanford_cars  # choose from ['stanford_cars', 'oxford_pets', 'oxford_flowers', 'fgvc_aircraft', 'dtd', 'eurosat', 'food101', 'sun397', 'caltech101', 'ucf101', 'imagenet', 'roxford5k', rparis6k']
--clip_model_name ViT-B/32  # choose your model type e.g. ViT-B/16 (clip), siglip-ViT-B-16, siglip2-ViT-B-16, SLIP-ViT-B-16
--query_exp_name clip_vitb14_proj  # Choose an experiment name
--project  # If set, then uses our (PCA) projection, original features otherwise
``` 
You should then get the mAP results printed.

Note: Code is adopted from ICLR 2025 OTI baseline.