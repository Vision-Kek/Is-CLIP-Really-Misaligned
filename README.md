# Reevaluating the Intra-Modal Misalignment Hypothesis in CLIP

📝 [Paper](https://arxiv.org/abs/2603.16100) accepted for CVPR 2026 | 🌐 [Project Page](https://vision-kek.github.io/Is-CLIP-Really-Misaligned/)

## Table of Contents
1. [Demonstrations](#demonstrations)
2. [Run Few-Shot Classification](#run-few-shot-classification)
3. [Run Image-to-Image Retrieval](#run-image-to-image-retrieval)

# Demonstrations

### Projection Minimal Example

With the following lines code, you can modify any image embedding and get the $PCA^\leftarrow$ results from the paper:
```python
import requests
import torch
import torch.nn.functional as F
import clip  # pip install git+https://github.com/openai/CLIP.git

clip_model, preprocess = clip.load("ViT-B/16")

# 1. Get ImageNet class names
response = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt')
class_names = response.text.strip().split('\n')

# 2. Get text features of class names
tokenized = clip.tokenize(class_names)
with torch.no_grad():
    text_embeds = F.normalize(clip_model.encode_text(tokenized), dim=1)
d = text_embeds.shape[1]

# 3. Get principal components
U,S,Vt = torch.linalg.svd(text_embeds.float())
selected_components = Vt[:d//2]  # keep d/2 (e.g. 256 of 512)

# 4. Now you can project any image embeddings
image_embeddings = torch.randn(100, 512)  # replace with your test image embeddings from clip_model.encode_image(...)
image_embeddings_proj = image_embeddings @ Vt[:d//2].T
print(image_embeddings_proj.shape)  # (100, 256)
```
Done. Now you can replace the original CLIP `image_embeddings` with the resulting `image_embeddings_proj` for better few-shot performance in classification and retrieval tasks.

### Intra-Modal Similarity Recovery

Following Appendix A, recover image-image similarities given only text-image similarities:
```python
import torch
import torch.nn.functional as F

N, d = 42, 3  # You can set the number of embeddings (N) and their dimension here (d)

# Setup
X_T = F.normalize(torch.rand(N,d),dim=1)  # text embeddings, hidden
X_I = F.normalize(torch.rand(N,d),dim=1)  # image embeddings, hidden
S_inter = X_T @ X_I.T  # text-image similarities, given

# Decompose
U, Sigma, Vt = torch.linalg.svd(S_inter)
U, V = U[:,:d], Vt.T[:,:d]  # reduced SVD (rank d)

pairwise_products = V.unsqueeze(2) * V.unsqueeze(1)

# Solve
A = pairwise_products.view(N,d*d)
b = torch.ones(N)

x = torch.linalg.lstsq(A, b, driver='gelsd').solution

# Recover
Q = x.view(d,d)
S_intra_recovered = V @ Q @ V.T

# Check
S_intra_true = X_I @ X_I.T
print('RESULT: recovery error:', (S_intra_recovered - S_intra_true).abs().max())
```

You can look into [s_intra_recovery.py](s_intra_recovery.py) for more analysis.
It walks you through the procedure of recovering $S_{intra}$ from $S_{inter}$ step by step.


## Run Few-Shot Classification

🚧 Soon!

## Run Retrieval

🚧 Soon!