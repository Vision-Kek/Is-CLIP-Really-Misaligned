import torch
from tqdm import tqdm
from types import SimpleNamespace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def fwd_texts(clip_model, tokenize_fn, texts):
    if 'CLIP' in str(type(clip_model)) or 'SLIP' in str(type(clip_model)):
        # as in clip
        tokenized = tokenize_fn(texts).to(device)
        chunk_size = 64
        text_feats = []
        for i in tqdm(range(1 + (len(tokenized)-1) // chunk_size), 'Forwarding Texts'):
            text_feats.append(clip_model.encode_text(tokenized[i * chunk_size:i * chunk_size + chunk_size]))
        return torch.cat(text_feats)
    else:
        # as in siglip
        tokenized = tokenize_fn(texts, padding="max_length", max_length=64, return_tensors="pt").to(device)
        return clip_model.encode_text(tokenized)
 
@torch.no_grad()
def fwd_imagenet_texts(clip_model, tokenize_fn, cache_path='imagenet_text_features.pt'):
    import os
    import json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'imagenet_classnames.json')) as f:
        imagenet_classes = json.load(f)
    if not os.path.exists(cache_path):
       text_feats = fwd_texts(clip_model, tokenize_fn, imagenet_classes)
       os.makedirs(os.path.dirname(cache_path), exist_ok=True)
       torch.save(text_feats, cache_path)
    return torch.load(cache_path)


def fit_pca(text_feats):
    n_components = text_feats.shape[1] // 2
    covmat = torch.cov(text_feats.T)
    eigvals, eigvecs = torch.linalg.eig(covmat)
    sorted_indcs = torch.sort(eigvals.float().cpu(), descending=True).indices
    return SimpleNamespace(components_=eigvecs[:, sorted_indcs[:n_components]].float().T)
    #from sklearn.decomposition import PCA
    #d = text_feats.shape[1]
    #pca = PCA(n_components=d//2).fit(text_feats.cpu())
    #return pca

def transform_pca(pca, feats_tensor):
    U = torch.tensor(pca.components_, device=feats_tensor.device, dtype=torch.float32) \
        if not isinstance(pca.components_, torch.Tensor) else pca.components_.to(feats_tensor.device) # (d/2, d)
    # U is the PCA projection form d -> d/2
    # then reproject into original space: d/2 -> d  (this step is technically not necessary, but conceptionally useful)
    feats_tensor_projected = (U.T @ U @ feats_tensor.T).T
    return feats_tensor_projected
