import torch
import os
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import SiglipImageProcessor, SiglipModel

MODEL_DIR = '/path/to/checkpoint/dir'
_MODELS = {
    "siglip2-ViT-B-16": os.path.join(MODEL_DIR, 'siglip2'),
    "siglip-ViT-B-16": os.path.join(MODEL_DIR, 'siglip')
}

# Needs to have a 'tokenizer' attribute that is accessed
class SigLIPWrapper:
    def __init__(self, model_name, device):
        self.model = SiglipModel.from_pretrained(_MODELS[model_name])
        self.model.eval()
        self.model.to(device)
        self.processor = SiglipImageProcessor.from_pretrained(_MODELS[model_name], do_convert_rgb=True)
        # inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")
        self.tokenizer = AutoTokenizer.from_pretrained(_MODELS[model_name])
        self.device = device

    @torch.no_grad()
    def encode_image(self, inputs): # expects tensor
        inputs = {k:v[0].to(self.device) for k,v in inputs.items()} # for some reason pixel_values is wrapped list
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
        return image_embeddings

    @torch.no_grad()
    def encode_text(self, inputs): # expects tensor
        #print(inputs.shape, 'text shape')
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features

def load_siglip(name, device = "cuda" if torch.cuda.is_available() else "cpu"):
    if name not in _MODELS:
        raise RuntimeError(f"Model {name} not found in available models: {list(_MODELS.keys())}")

    wrapper = SigLIPWrapper(name, device)
    return wrapper, wrapper.processor