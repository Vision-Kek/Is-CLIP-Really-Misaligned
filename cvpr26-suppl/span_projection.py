import torch
from tqdm import tqdm
import os
import requests

def fit_pca(feats, reduce_factor=2):
    n_components = feats.shape[1] // reduce_factor
    covmat = torch.cov(feats.T)
    eigvals, eigvecs = torch.linalg.eigh(covmat)
    sorted_indcs = torch.sort(eigvals.float().cpu(), descending=True).indices
    return eigvecs[:, sorted_indcs[:n_components]].float().T

def transform_pca(span, feats_tensor):
    U = torch.tensor(span, device=feats_tensor.device, dtype=feats_tensor.device) \
        if not isinstance(span, torch.Tensor) else span.to(feats_tensor.device, feats_tensor.dtype)  # (d/2, d)
    # U is the PCA projection form d -> d/2
    # then reproject into original space: d/2 -> d  (this step is technically not necessary, but conceptionally useful)
    feats_tensor_projected = (U.T @ U @ feats_tensor.T).T
    return feats_tensor_projected


class SpanProjection:
    def __init__(self, clip_model, tokenize_fn, span_feats_cache='imagenet_text_features.pt', device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not device else device
        self.clip_model = clip_model
        self.tokenize_fn = tokenize_fn
        self.span_feats_cache = span_feats_cache
        self.fwd_texts = self.fwd_texts_clip \
            if 'CLIP' in str(type(self.clip_model)) or 'SLIP' in str(type(self.clip_model)) \
            else self.fwd_texts_siglip

        self.fullspan = self.fwd_imagenet_texts()
        self.span = fit_pca(self.fullspan)

    @torch.no_grad()
    def fwd_texts_clip(self, texts):
        # as in clip
        tokenized = self.tokenize_fn(texts).to(self.device)
        chunk_size = 64
        text_feats = []
        for i in tqdm(range(1 + (len(tokenized) - 1) // chunk_size), 'Forwarding Texts'):
            text_feats.append(self.clip_model.encode_text(tokenized[i * chunk_size:i * chunk_size + chunk_size]))
        return torch.cat(text_feats)

    @torch.no_grad()
    def fwd_texts_siglip(self, texts):
        # as in siglip
        tokenized = self.tokenize_fn(texts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
        return self.clip_model.encode_text(tokenized)

    @torch.no_grad()
    def fwd_imagenet_texts(self):
        if not os.path.exists(self.span_feats_cache):
            print(f'{self.span_feats_cache=} not exists, extracting text features')
            response = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt')
            imagenet_classes = response.text.strip().split('\n')
            text_feats = self.fwd_texts(imagenet_classes)
            os.makedirs(os.path.dirname(self.span_feats_cache), exist_ok=True)
            torch.save(text_feats, self.span_feats_cache)
        return torch.load(self.span_feats_cache, map_location=self.device).to(torch.float32)

    @torch.no_grad()
    def project(self, features):
        return transform_pca(self.span, features)