import torch
import numpy as np
import tqdm
from PIL import Image
import pandas as pd
from torchvision.transforms  import ToTensor
from skimage import metrics
from networks import load_network
from utils import read_config, get_atk_dir_name
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import warnings
warnings.filterwarnings("ignore")

def compute_metrics(attack, models, list_net_at, list_images, dataset_dir, adv_dir, out_dir, device='cuda'):
    pil2torch = ToTensor()
    with torch.no_grad():
        print(attack)
        for namenetA in [None, ] + list_net_at:
            for label, class_name in enumerate(['real', 'fake']):
                if namenetA is None:
                    key = 'ori%d' % label
                else:
                    key = 'adv%d_%s' % (label, namenetA)
                list_images.loc[:, 'psnr_'+key] = float(np.nan)
                list_images.loc[:, 'ssim_'+key] = float(np.nan)
                for model in models:
                    list_images.loc[:, model+'_'+key] = float(np.nan)
            list_images = list_images.copy()

        for index in tqdm.tqdm(list_images.index):
            for label in range(2):
                ori_filename = Path(dataset_dir, list_images.loc[index,f'filename{label}'])
                ori_img = np.asarray(Image.open(ori_filename))
                for namenetA in [None, ] + list_net_at:
                    if namenetA is None:
                        ori_filename = Path(dataset_dir, list_images.loc[index,f'filename{label}'])
                        img = Image.open(ori_filename)
                        key = 'ori%d' % label
                    else:
                        filename = Path(adv_dir, 'png', namenetA, attack, f'adv{label}', f"{list_images.loc[index,'name']}.png")
                        img = Image.open(filename)
                        key = 'adv%d_%s' % (label, namenetA)
                    torch_img = pil2torch(img).to(device)[None,:]
                    numpy_img = np.asarray(img)
                    list_images.loc[index, 'psnr_'+key] = metrics.peak_signal_noise_ratio(ori_img, numpy_img, data_range=255.0)
                    list_images.loc[index, 'ssim_'+key] = metrics.structural_similarity(ori_img, numpy_img, channel_axis=-1, data_range=255.0)
                    for model in models:
                        out = torch.diff(models[model](torch_img)[0]).item()
                        list_images.loc[index, model+'_'+key] = float(out)
                        
        list_images.to_csv(out_dir / f'res_{attack}.csv', index=False)

def get_results(res_csv, models):
    tab = pd.read_csv(res_csv)
    models_att = models

    out_tab = pd.DataFrame(columns=['model_att','psnr','ssim',]+models)
    for mm in ['ori', ] + models_att:
        if mm=='ori':
            key = ['ori%d' % (classes_id) for classes_id in [0,1]]
            out_tab.loc[mm,'model_att'] = 'original'
        else:
            key = ['adv%d_%s' % (classes_id,mm) for classes_id in [0,1]]
            out_tab.loc[mm,'model_att'] = mm
        out_tab.loc[mm,'psnr'] = (tab.loc[:,'psnr_'+key[0]].clip(0,55).mean() + tab.loc[:,'psnr_'+key[1]].clip(0,55).mean())/2
        out_tab.loc[mm,'ssim'] = (tab.loc[:,'ssim_'+key[0]].mean() + tab.loc[:,'ssim_'+key[1]].mean())/2

        for namenet in models:
            out_tab.loc[mm,namenet] = ((tab.loc[:,namenet+'_'+key[0]]>0).mean() + (tab.loc[:,namenet+'_'+key[1]]<=0).mean())/2

    return out_tab

def plot_heatmap(res_df, models):
    d = 100*res_df.loc[models,models]
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(data=d.apply(pd.to_numeric), clim=(0,100), cmap='Reds', annot=True, fmt=".1f", ax=ax)
    plt.axis('image')

    ax.set_xticklabels(models, rotation=90)
    ax.set_yticklabels(models, rotation=0)
    #plt.savefig(f'figure_{method_att}.svg', bbox_inches='tight', pad_inches=0.0)

def main(config_yaml: str, img_csv: str, dataset_dir: str, adv_dir: str, res_dir: str, device='cuda'):
    config = read_config(config_yaml)
    
    list_net = [net['name'] for net in config['networks']]
    list_net_at = list_net
    models = {namenet: load_network(namenet).to(device).eval() for namenet in list_net}

    attacks = [get_atk_dir_name(atk['name'], atk['args']) for atk in config['attacks']]

    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    list_images = pd.read_csv(img_csv)
    for attack in attacks:
        if not Path(res_dir, f'res_{attack}.csv').exists():
            compute_metrics(attack, models, list_net_at, list_images, dataset_dir, adv_dir, res_dir, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metrics.')

    parser.add_argument('--config', type=str, default='./configs/latent.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--img_csv', type=str, default='./AdversarialRobustnessCLIP/latent/list_images_latent.csv',
                        help='Path to the CSV file containing image list')
    parser.add_argument('--dataset_dir', type=str, default='./AdversarialRobustnessCLIP',
                        help='Directory containing the dataset')
    parser.add_argument('--adv_dir', type=str, default='./images/latent',
                        help='Directory to save adversarial images')
    parser.add_argument('--res_dir', type=str, default='./res_csv/latent',
                        help='Directory to save results')

    args = parser.parse_args()
    main(args.config, args.img_csv, args.dataset_dir, args.adv_dir, args.res_dir)