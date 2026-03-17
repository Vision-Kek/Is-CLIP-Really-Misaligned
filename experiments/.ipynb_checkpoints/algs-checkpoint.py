import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm

def calc_simmat(embs):
    embs = F.normalize(embs)
    simmat = embs @ embs.T
    return simmat

def fs_classification(vembs, labels, shots=1):
    preds = []
    for etest, label in zip(vembs, labels):
        support_cat = torch.randint(high=12499, size=(shots,))
        support_cat = F.normalize(vembs[support_cat].mean(dim=0,keepdim=True))
        support_dog = torch.randint(low=12499, high=24998, size=(shots,))
        support_dog = F.normalize(vembs[support_dog].mean(dim=0,keepdim=True))
        preds.append(torch.argmax(etest@torch.cat([support_cat, support_dog]).T, dim=-1))
    acc = (torch.stack(preds) == torch.tensor(labels)).float().mean()
    return acc

def eval_fs_classification(embs, labels, shots=[1]):
    res=[]
    for _shots in shots:
        res.append(fs_classification(embs, labels, _shots))
    return res

def measure_retrieval_map(features, labels):
    """
    Standard evaluation: each query vs all others (excluding itself)
    
    Args:
        features: numpy array of shape (n_samples, feature_dim)
        labels: array-like of shape (n_samples,) 
    
    Returns:
        mAP: mean Average Precision
    """
    labels = torch.tensor(labels, device=features.device)
    features = features / features.norm(dim=-1, keepdim=True)
    similarity = features @ features.T  # (n_samples, n_samples)
    # Set self-similarity to -1 to exclude query from results
    similarity.fill_diagonal_(-1.)
    
    APs = []
    for i in tqdm(range(len(features))):
        relevance = (labels == labels[i]).float() # Binary relevance
        relevance[i] = 0  # Exclude query itself
        
        sorted_indices = torch.argsort(similarity[i], descending=True)
        sorted_relevance = relevance[sorted_indices]
        
        # Compute AP only if there are relevant items
        if relevance.sum() > 0:
            ap = average_precision_score(relevance.cpu(), similarity[i].cpu())
            APs.append(ap)
    
    return torch.tensor(APs).mean()

def measure_text_to_image_retrieval_map(image_feats, text_feats, labels):
    labels = torch.tensor(labels, device=image_feats.device)
    assert len(text_feats)==len(labels.unique())
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    similarity = image_feats @ text_feats.T  # (n_imgs, n_classes)

    APs = []
    for _class in range(similarity.shape[1]):
        relevance = (labels == _class).float() # Binary relevance
        ap = average_precision_score(relevance.cpu(), similarity[:, _class].cpu())
        APs.append(ap)
    return torch.tensor(APs).mean()

def measure_retrieval_top1_map(embs):
    """ measure top 1 retrieval """
    embs = embs.to(device)
    pred = []
    for i,query in tqdm(enumerate(embs)):
        gallery = embs[[j for j in range(len(embs)) if j!=i ]]
        retrieved = (query @ gallery.T).argmax()
        pred = dataset.labels[retrieved]
    _map = (torch.tensor(pred).int().cpu() == torch.tensor(dataset.labels)).float().mean()
    return _map