from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CatDogDataset(Dataset):
    def __init__(self, dir='/mnt/external/datasets/PetImages'):
        self.catdog_dataset = dir
        self.categ_names = ['Cat', 'Dog'] 
        self.img_pths = []
        self.labels = []
        for categ_idx, categ_name in enumerate(self.categ_names):
            f_img = os.listdir(os.path.join(self.catdog_dataset, categ_name))
            for f in f_img: 
                f = os.path.join(self.catdog_dataset, categ_name, f)
                if not f.endswith('.jpg') or os.path.getsize(f) == 0: continue
                self.img_pths.append(f)
                self.labels.append(categ_idx)
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return Image.open(self.img_pths[idx]), self.labels[idx]
