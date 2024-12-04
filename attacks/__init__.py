'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import torchattacks
from .robust_fgsm.attack_images import attack_images as robust_attack
from .pgd.attack_images import attack_images as pgd
from .universal.search import universal
import inspect
from utils import torch2pil

def select_attack_dict(model, my_atk: dict):
    name = my_atk['name']
    args = my_atk['args'].copy()

    atk_dict = {name: cls for name, cls in inspect.getmembers(torchattacks, inspect.isclass)}
    if name not in atk_dict:
        raise ValueError(f"The attack '{name}' is not supported. Choose from: {list(atk_dict.keys())}")

    if 'eps' in args:
        args['eps'] = args['eps']/255
    if 'alpha' in args:
        args['alpha'] = args['alpha']/255
    if 'max_epsilon' in args:
        args['max_epsilon'] = args['max_epsilon']/255

    return atk_dict[name](model, **args)

def attack_images(atk, label, img_csv, dataset_dir, adv_img_dir, noise_dir):
    pil2torch = ToTensor()
    adv_img_dir = Path(adv_img_dir, f'adv{label}')
    adv_img_dir.mkdir(parents=True, exist_ok=True)
    noise_dir = Path(noise_dir, f'adv{label}')
    noise_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(img_csv)
    img_paths = df[f'filename{label}'].apply(lambda x: Path(dataset_dir, x))

    labels = torch.ones((1,)).long() if label == 1 else torch.zeros((1,)).long()
    atk.set_mode_targeted_by_function(target_map_function=lambda images, labels:(labels+1)%2)
    
    for img_path in tqdm(img_paths):
        adv_path = adv_img_dir / img_path.name
        if not adv_path.exists():
            ori_img = pil2torch(Image.open(img_path))
            images = ori_img[None, :]

            #Save adv. image
            adv_img = atk(images, labels)[0].cpu()
            torch2pil(adv_img).save(adv_path)

            #Save noise
            noise = (adv_img - ori_img).cpu().numpy()
            noise = np.transpose(noise, (1, 2, 0))
            np.save(noise_dir / adv_path.with_suffix('.npy').name, noise)


