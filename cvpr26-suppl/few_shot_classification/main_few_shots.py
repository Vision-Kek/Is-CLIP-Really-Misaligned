import os
import pickle
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
from utils import *
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import span_projection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', choices=['prototype', 'gda'], default='gda')
    parser.add_argument('--project', action='store_true')
    parser.add_argument('--use_text', action='store_true')
    parser.add_argument('--data_root_path', default='datasets')
    parser.add_argument('--dataset', choices=[
        'stanford_cars', 'oxford_pets', 'oxford_flowers', 'fgvc_aircraft',
        'dtd', 'eurosat', 'food-101', 'caltech101', 'ucf101', 'sun397', 'imagenet'
    ], required=True)
    parser.add_argument('--backbone', choices=['RN50', 'ViT-B/16', 'ViT-B/32', 'ViT-L/14',
                                               'siglip-ViT-B-16', 'siglip2-ViT-B-16',
                                               'dinov2_vitb14','dinov3_vitl16'], default='RN50')
    args = parser.parse_args()
    args.subsample_classes = 'all' #  dataset classes need it
    return args

def calc_acc_from_sims(sims, test_labels):
    test_logits = sims / 0.01
    notune_acc = cls_acc(test_logits, test_labels)
    return notune_acc

############## CLASSIFIERS ##############

def prototype_classify(inputs):
    class_protos = torch.stack([
        inputs['train_features'][inputs['train_labels']==i].mean(dim=0)
        for i in inputs['train_labels'].unique()
    ]) # shape (n,d)
    class_protos = F.normalize(class_protos, dim=-1).float()

    fs_sims = inputs['test_features'].float() @ class_protos.T  # (N,n)

    if inputs.get('clip_weights') is not None:
        # text use
        txt_sims = inputs['test_features'].float() @ inputs['clip_weights'].T
        alpha = 0.5
        sims = (1-alpha)*fs_sims + alpha*txt_sims
    else:
        # no text use
        sims = fs_sims
    return calc_acc_from_sims(sims, inputs['test_labels'])

def zero_shot_classify(inputs):
    assert inputs.get('clip_weights') is not None, '"zero-shot" means using the text embs as weights, u gotta pass them'
    sims = inputs['test_features'].float() @ inputs['clip_weights'].float()
    return calc_acc_from_sims(sims, inputs['test_labels'])

def lda_iclr24_classify(inputs):
    vecs, labels = inputs['train_features'].float(), inputs['train_labels'].float()

    n_class = len(labels.unique())
    # Parameter Estimation.
    # normal distribution
    mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(n_class)])

    # KS Estimator.
    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in range(n_class)])
    cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov()
                + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).to(device))

    ps = torch.ones(n_class).to(device) * 1. / n_class

    W = torch.einsum('nd, dc -> cn', mus, cov_inv)
    b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2

    if inputs.get('clip_weights') is not None:
        alpha = 0.5 # no validation set available in our setting, can't do grid search
        sims = ((1-alpha) * inputs['test_features'].float() @ inputs['clip_weights'].float()
                + alpha * (inputs['test_features'].float() @ W + b))
    else:
        sims = inputs['test_features'].float() @ W + b
    return calc_acc_from_sims(sims, inputs['test_labels'])

############ DATALOADERS ############

def get_train_val_test_features(cfg, all_feats, image_paths, clip_model=None, dataset=None):
    if dataset is None:
        assert cfg['dataset'] != "imagenet"
        dataset = build_dataset(cfg, dataset=cfg['dataset'].replace('-', ''), root_path=cfg["data_root_path"], shots=cfg['shots'])

        # cached: don't load actual image, just path
        train_loader = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm='cached', is_train=True,
                                               shuffle=False)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm='cached',
                                        shuffle=False)
        val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm='cached',
                                       shuffle=False)

        val_features, val_labels = load_preextracted_features(all_feats, image_paths, val_loader, device)
    else:
        # imagenet
        dataset.set_fewshot_samples(cfg['shots'])  # random sampling is in set_fewshot_samples function
        # train
        train_loader = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)

        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)

        val_features, val_labels = None, None

    # load pre-extracted image features
    train_features, train_labels = load_preextracted_features(all_feats, image_paths, train_loader, device)
    test_features, test_labels = load_preextracted_features(all_feats, image_paths, test_loader, device)

    return {
        'train_features': train_features,
        'train_labels': train_labels,
        'test_features': test_features,
        'test_labels': test_labels,
        'val_features': val_features, # unused
        'val_labels': val_labels, # unused
    }

############ MAIN ############

def main():
    args = get_arguments()
    
    cfg = args.__dict__
    
    # Load cfg for conditional prompt.
    print(f"\nRunning config\n:{cfg}\n")

    bb = cfg["backbone"].replace("/", "_")
    if not 'siglip' in bb.lower() and not 'SLIP' in bb.lower() and not 'dino' in bb.lower():
        clip_model, preprocess = clip.load(cfg['backbone'])
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
    else:
        clip_model = None
         # tokenize_fn=

    dtype = torch.float32
    # pre-extracted image embeddings
    feat_dir = os.path.join('data', 'image_features', cfg["dataset"])
    all_feats = torch.load(f'{feat_dir}/{cfg["backbone"].replace("/", "_")}.pt').to(dtype)

    if args.use_text:
        # pre-extracted text embeddings, only needed when blending text classifier weights
        text_feats = torch.load(os.path.join('data', 'text_features', cfg["dataset"], f'{bb}.pt')).to(dtype)

    # if Imagenet, load only once, takes too long otherwise
    dataset = ImageNet(cfg, cfg['data_root_path'], preprocess=None, cached=True) if cfg['dataset'] == "imagenet" else None

    with open(f'{feat_dir}/image_names.pkl', 'rb') as f:
        all_img_pths = pickle.load(f)

    classifier_registry = {
        'prototype': prototype_classify,
        'gda': lda_iclr24_classify,
    }
    classifier = classifier_registry.get(args.classifier)
    assert classifier, f'Unknown classifier {args.classifier}, choose from {classifier_registry.keys()}'

    if args.project:
        if 'dino' in cfg['backbone']:
            print('dino has no text encoder, so we cant run projection, aborting')
            return
        imagenet_text_feats_dir = os.path.join('data', 'text_features', "imagenet", f'{bb}.pt')
        print(f'Loading imagenet text span from {imagenet_text_feats_dir}')
        projection = span_projection.SpanProjection(clip_model, tokenize_fn=None,
                                                    span_feats_cache=imagenet_text_feats_dir, device=None)

    seeds = [1, 2, 3]
    shots = [1, 2, 4, 8, 16]
    notune_accs = {s:[] for s in seeds}
    
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)

        for shot in shots:
            cfg["shots"] = shot

            input_features = get_train_val_test_features(cfg, all_feats, all_img_pths, dataset=dataset, clip_model=clip_model)

            if args.use_text:
                input_features['clip_weights'] = text_feats

            if args.project:
                input_features['train_features'] = projection.project(input_features['train_features'])
                input_features['test_features'] = projection.project(input_features['test_features'])

            acc = classifier(input_features)  # Run

            notune_accs[seed].append(acc)
            print(f"{seed=}, {shot=}, {acc=:.2f}")

    print({k:v for k,v in cfg.items() if k!='classnames'})
    print(f"Evaluate on {seeds=}")
    print(f"Evaluate on {shots=}")
    notune = []
    for seed in seeds:
        print(f"{seed=}", notune_accs[seed])
        notune.append(notune_accs[seed])
    print("Average: ", torch.tensor(notune).mean(dim=0))
    
if __name__ == '__main__':
    main()