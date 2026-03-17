import os
import argparse

import torch

from datasets.imagenet import ImageNet
from datasets import build_dataset
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=[
        'stanford_cars', 'oxford_pets', 'oxford_flowers', 'fgvc_aircraft',
        'dtd', 'eurosat', 'food-101', 'caltech101', 'ucf101', 'sun397', 'imagenet'
    ], required=True)
    parser.add_argument('--data_root_path', default='datasets')
    parser.add_argument('--subsample_classes', default='all')
    parser.add_argument('--backbone', choices=['RN50', 'ViT-B/16', 'ViT-B/32'], default='RN50')
    args = parser.parse_args()
    return args


def main():
    assert 'data' in os.listdir(os.getcwd()), 'data/ not found. Run from parent dir with python -m scripts/preextract'
    # Load config file
    args = get_arguments()
    cfg = args.__dict__

    # overwrite backbone
    cfg['backbone'] = args.backbone
    cfg['shots'] = 0  # dummy, doesn't matter for text extraction

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    if cfg['dataset'] != "imagenet":
        dataset = build_dataset(cfg, dataset=cfg['dataset'].replace('-', ''), root_path=cfg["data_root_path"],
                                shots=cfg['shots'])
    else:
        dataset = ImageNet(cfg, cfg['data_root_path'], preprocess=None, cached=True)
        
    text_features = clip_classifier(dataset.classnames, dataset.template, clip_model).T

    # Save extracted features
    save_dir = f'data/text_features/{cfg["dataset"]}'
    print(text_features.shape)
    print('Writing to', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(text_features, f'{save_dir}/{cfg["backbone"].replace("/", "_")}.pt')
    print('Done.')

if __name__ == '__main__':
    main()