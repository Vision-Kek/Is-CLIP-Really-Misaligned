import torch
import os
import torchvision.transforms as T

MODEL_DIR = '/path/to/checkpoint/dir' # replace
DINOV2_DIR = '/path/to/dinov2/repo/' # replace, git clone from facebook
DINOV3_DIR = '/path/to/dinov3/repo/' # replace, git clone from facebook
_MODELS = {
    "dinov2_vitb14": os.path.join(MODEL_DIR, 'dinov2', "dinov2_vitb14_reg4_pretrain.pth"),
    "DINOv2-ViT-L-14": os.path.join(MODEL_DIR, "dinov2_vitl14_reg4_pretrain.pth"),
    "dinov3_vitl16": os.path.join(MODEL_DIR, 'dinov3', "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    "DINOv3.txt-ViT-L-16": os.path.join(MODEL_DIR, 'dinov3', "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"),
}


from SLIP import transform as preprocess_transform

class DINOWrapper:
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.preprocess = preprocess_transform

    @torch.no_grad()
    def encode_image(self, image): # expects tensor
        if image.ndim==3:
            # add batch dim
            inputs = image.unsqueeze(0)
        feats = self.model.forward_features(image)
        return feats['x_norm_clstoken']

def setup_dinov3(name, device):
    model = torch.hub.load(repo_or_dir=DINOV3_DIR,
                                      source='local',
                                      model=name,
                                      weights=_MODELS[name])
    wrapper = DINOWrapper(model, device)
    return wrapper, wrapper.preprocess

def setup_dinov2(name, device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=False)
    model.load_state_dict(torch.load(_MODELS[name]))
    # model = torch.hub.load(repo_or_dir=DINOV2_DIR,
    #                                   source='local',
    #                                   model=name,
    #                                   weights=_MODELS[name])
    wrapper = DINOWrapper(model, device)
    return wrapper, wrapper.preprocess

def load_dino(name, device = "cuda" if torch.cuda.is_available() else "cpu"):
    if name not in _MODELS:
        raise RuntimeError(f"Model {name} not found in available models: {list(_MODELS.keys())}")

    if 'v3' in name:
        return setup_dinov3(name, device)
    elif 'v2' in name:
        return setup_dinov2(name, device)
