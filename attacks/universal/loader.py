'''
from: https://github.com/qilong-zhang/Pytorch_Universal-adversarial-perturbation
'''

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision.transforms import ToTensor
from pathlib import Path

def get_train_test_data(csv_path: str, label, dataset_dir = './dataset', out_dir = '.', test_split = 0.2):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)

    img_paths = list(df[f'filename{label}'].apply(lambda x: Path(dataset_dir, x)))
    train_imgs = img_paths[:-int(len(img_paths)*test_split)]
    test_imgs = img_paths[-int(len(img_paths)*test_split):]
    train_df = pd.DataFrame(train_imgs, columns=["filename"])
    #train_df.to_csv(out_dir / f'train_{label}.csv', index=False)
    test_df = pd.DataFrame(test_imgs, columns=["filename"])
    #test_df.to_csv(out_dir / f'test_{label}.csv', index=False)

    return train_df, test_df

class AtkDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.img_df = df
        self.transforms = transforms
        self.pil2torch = ToTensor()

    def __getitem__(self, index):
        img = self.pil2torch(Image.open(self.img_df.loc[index, f'filename']))
        if self.transforms:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.img_df)