'''
from: https://github.com/qilong-zhang/Pytorch_Universal-adversarial-perturbation
'''

import warnings
warnings.filterwarnings("ignore")

from .universal_pert import universal_perturbation
from .loader import AtkDataset, get_train_test_data
from .save_pert_images import save_pert_images
from pathlib import Path

def universal(namenet, attack, label, img_csv, dataset_dir, adv_img_dir, noise_dir):
    args = attack['args']
 
    pert_dir = Path(noise_dir, f'adv{label}')
    pert_dir.mkdir(parents=True, exist_ok=True)
    
    #Load Data
    train_df, test_df = get_train_test_data(img_csv, label, dataset_dir)
    train_data = AtkDataset(train_df)
    val_data = AtkDataset(test_df)

    #Compute universal pert
    try:
        v = universal_perturbation(dataset = train_data, valset = val_data, \
                                    namenet = namenet, out_dir = pert_dir, **args)
        
        adv_img_dir = Path(adv_img_dir, f'adv{label}')
        adv_img_dir.mkdir(parents=True, exist_ok=True)
        v = v.squeeze().transpose(1, 2, 0)
        print('Saving adv. images...')
        save_pert_images(v, img_csv, dataset_dir, label, adv_img_dir)
        
    except Exception as e:
        print(e)