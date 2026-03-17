import torch
from tqdm import tqdm
import clip

def loop_dinov2_l(dataloader):
    from fs_predicate_classification.models import dinov2_txt
    dino_embs = []
    for ep in tqdm(dataloader):
        imgs = [x[0].convert('RGB') for x in ep]
        with torch.no_grad():
            dinores = dinov2_txt.Inference(imgs, grid_size=(16,16))
        dino_embs.append(dinores.backbone_class_tokens.cpu())
    embs = torch.cat(dino_embs)
    return embs

def loop_dinov3_l(dataloader):
    from fs_predicate_classification.models import dinov3_txt
    dino_embs = []
    for ep in tqdm(dataloader):
        imgs = [x[0].convert('RGB') for x in ep]
        with torch.no_grad():
            dinores = dinov3_txt.Inference(imgs, grid_size=(16,16))
        dino_embs.append(dinores.backbone_class_tokens.cpu())
    embs = torch.cat(dino_embs)
    return embs

class DinoHF:
    def __init__(self, device='cuda'):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

    def dinov2_hf_fwd(self, imgbatch):
       with torch.no_grad():
            inputs = self.processor(images=imgbatch, return_tensors="pt", use_fast=True).to(self.device)
            outputs = self.model(**inputs)
    
       last_hidden_states = outputs[0]
       cls_token = last_hidden_states[:, 0, :]
       return cls_token
    
    def loop(self, dataloader):
        dino_b_embs = []
        for batch in tqdm(dataloader):
            imgs = [x[0].convert('RGB') for x in batch]
            cls_emb = self.dinov2_hf_fwd(imgs)
            dino_b_embs.append(cls_emb.cpu())
        return torch.cat(dino_b_embs)


def loop_clip(clipmodel, clippreprocess, dataloader, device='cuda'):
    vembs = []
    for ep in tqdm(dataloader):
        imgs = [x[0] for x in ep]
        with torch.no_grad():
            batch = torch.stack([clippreprocess(img) for img in imgs]).to(device)
            vemb = clipmodel.encode_image(batch)
        vembs.append(vemb.cpu())
    vembs = torch.cat(vembs)
    return vembs
