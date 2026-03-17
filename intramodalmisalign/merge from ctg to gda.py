import torch
import pickle
import os


def merge(dataset, model):
    # 1. Keep image paths in `imagenames.pkl`
    gda_n = pickle.load(open(f'/home/jonas/workspace/ICLR24/data/image_features/{dataset}/image_names.pkl','rb'))
    n_total = len(gda_n)
    # 2. Merge train-> <-test *embeds* from CTG into 1 tensor
    ctg_train_x = torch.load(f'/home/jonas/workspace/Cross-the-Gap/data/image_features/{dataset.replace("-","")}/{model}/train/image_features.pt')
    ctg_test_x = torch.load(f'/home/jonas/workspace/Cross-the-Gap/data/image_features/{dataset.replace("-","")}/{model}/test/image_features.pt')
    n_train = len(ctg_train_x)
    n_test = len(ctg_test_x)
    n_val = n_total - n_train - n_test
    print(n_train, n_val, n_test)
    # 3. Insert zeros for *embeds* that are missing in the center (correspond to val)
    merged = torch.cat([ctg_train_x, torch.zeros(n_val, ctg_train_x.shape[1]), ctg_test_x])
    return merged


all_datasets=  os.listdir(f'/home/jonas/workspace/ICLR24/data/image_features/')
model = 'dinov2_vitb14'

merged = merge('sun397', model)
gda_out_file = f'/home/jonas/workspace/ICLR24/data/image_features/{'sun397'}/{model}.pt'
print('sun397', merged.shape)
torch.save(merged, gda_out_file)

def iter_datasets(all_datasets):
    for dataset in all_datasets:
        try:
            merged = merge(dataset, model)
        except Exception as e:
            print(e)
        gda_out_file = f'/home/jonas/workspace/ICLR24/data/image_features/{dataset}/{model}.pt'
        print(dataset, merged.shape)
        torch.save(merged, gda_out_file)