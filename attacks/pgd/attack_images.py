import torch
from .pgd import PGDLmask, PGDmask
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
import pandas as pd
from utils import torch2pil
from pathlib import Path

def pgd_attack(model, my_atk: dict):
    atk_name = my_atk['name']
    args = my_atk['args']
    eps = args['eps']
    steps = args['steps']

    if atk_name == 'pgdE':
        atk = PGDLmask(model, eps=eps, alpha=eps/5.0, steps=steps, p=2)
    elif atk_name == 'pgdI':
        atk = PGDmask(model, eps=eps/255.0, alpha=eps/255.0/4.0, steps=steps)
    else:
        raise ValueError(f"The attack '{atk_name}' is not supported.")
    return atk

def attack_images(model, atk_dict, label, img_csv, dataset_dir, adv_img_dir, noise_dir, device='cuda'):
    pil2torch = ToTensor()
    adv_img_dir = Path(adv_img_dir, f'adv{label}')
    adv_img_dir.mkdir(parents=True, exist_ok=True)
    noise_dir = Path(noise_dir, f'adv{label}')
    noise_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(img_csv)
    img_paths = df[f'filename{label}'].apply(lambda x: Path(dataset_dir, x))

    labels = torch.ones((1,)).long() if label == 1 else torch.zeros((1,)).long()
    atk = pgd_attack(model, atk_dict)
    atk.set_mode_targeted_by_function(target_map_function=lambda images, labels:(labels+1)%2)

    for img_path in tqdm(img_paths):
        adv_path = adv_img_dir / img_path.name
        if not adv_path.exists():
            ori_img = pil2torch(Image.open(img_path))
            adv_nsd = atk.get_adv_noise(ori_img[None,:].to(device), labels)[0].cpu()
            #Save noise
            np.save(noise_dir / adv_path.with_suffix('.npy').name, adv_nsd.numpy().transpose(1, 2, 0))
            #Save adv. image
            torch2pil(ori_img+adv_nsd).save(adv_path)


