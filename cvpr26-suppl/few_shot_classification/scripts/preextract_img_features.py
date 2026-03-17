import os
import random
import argparse
import yaml
from tensorboard.compat.tensorflow_stub.io.gfile import exists
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_tranform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument('--backbone', choices=['RN50', 'ViT-B/16', 'ViT-B/32'], default='RN50')
    args = parser.parse_args()
    return args


def extract_features(clip_model, loader, norm=True, imagenet=False):
    features, image_paths = [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            if imagenet: images, imgpt = batch[0].cuda(), batch[1].cuda()
            else: images, imgpt = batch['img'].cuda(), batch['impath']
            image_features = clip_model.encode_image(images)
            if norm:
                image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            image_paths.extend(imgpt)

    features, labels = torch.cat(features), image_paths
    return features, labels


def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    assert 'data' in os.listdir(os.getcwd()), 'data/ not found. Run from parent dir with python -m scripts/preextract'

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # overwrite backbone
    cfg['backbone'] = args.backbone

    # Load cfg for conditional prompt.
    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    import pickle
    print("Preparing dataset.")
    if cfg['dataset'] != "imagenet":
        dataset = build_dataset(cfg, cfg['dataset'], cfg['root_path'], shots=-1) # -1: all

        train_loader = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform,
                                               is_train=True, shuffle=False)

        test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False,
                                        tfm=preprocess, shuffle=False)
        val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess,
                                       shuffle=False)

        print('Forwarding Train')
        train_features, train_paths = extract_features(clip_model, train_loader)
        print('Forwarding Val')
        val_features, val_paths = extract_features(clip_model, val_loader)
        print('Forwarding Test')
        test_features, test_paths = extract_features(clip_model, test_loader)

        print('train, val, test')
        print(len(train_paths), len(val_paths), len(test_paths))
        print(train_features.shape, val_features.shape, test_features.shape)

        all_feats = torch.cat([train_features, val_features, test_features])
        all_paths = train_paths + val_paths + test_paths


    else:
        dataset = ImageNet(cfg, cfg['root_path'], num_shots=-1, preprocess=preprocess) # -1: all
        # train
        train_loader = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8,
                                                         shuffle=False)

        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)

        print('Forwarding Train')
        train_features, train_paths = extract_features(clip_model, train_loader, imagenet=True)
        print('Forwarding Test')
        test_features, test_paths = extract_features(clip_model, test_loader, imagenet=True)

        print('train, val, test')
        print(train_features.shape, test_features.shape)

        all_feats = torch.cat([train_features, test_features])
        all_paths = train_paths  + test_paths

    # Save extracted features and image paths
    save_dir = f'data/image_features/{cfg["dataset"]}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(all_feats, f'{save_dir}/{cfg["backbone"].replace("/","_")}.pt')
    with open(f'{save_dir}/image_names.pkl', 'wb') as f:
        pickle.dump(all_paths, f)


if __name__ == '__main__':
    main()