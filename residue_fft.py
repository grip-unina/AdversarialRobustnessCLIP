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
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from utils import read_config, get_atk_dir_name
import argparse

def save_fft_uni(atk_name, list_net, adv_dir, out_dir):
    out_dir = Path(out_dir, atk_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for namenetA in list_net:
        out_path = Path(out_dir, namenetA + '_ffte.png') 
        print(atk_name, namenetA, flush=True)
        if not out_path.exists():
            paths = []
            for label in range(2):
                noise_dir = Path(adv_dir, 'noise', namenetA, atk_name, f'adv{label}')
                noises = list(noise_dir.glob('*.npy'))
                assert len(noises) > 0, f'No adv. noise  in {noise_dir}'
                paths.append(max(noises, key=lambda x: float(x.stem)))

            res = (np.load(paths[0]) + np.load(paths[1]))/2

            rey = np.abs(np.fft.fftshift(np.fft.fftn(res, axes=(0,1)), axes=(0,1)))**2
            rey = np.mean(rey,-1)
            energy2 = np.mean(rey)
            rey = rey / 4 / energy2

            plt.imsave(out_path,  rey.clip(0, 1), vmin=0, vmax=1)

def save_fft(atk_name, list_net, img_df, adv_dir, out_dir):
    out_dir = Path(out_dir, atk_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for namenetA in list_net:
        out_path = Path(out_dir, namenetA + '_ffte.png')
        print(atk_name, namenetA, flush=True)
        if not out_path.exists():
            list_noise = list()
            for label in range(2):
                noise_dir = Path(adv_dir, 'noise', namenetA, atk_name, f'adv{label}')
                assert len(img_df) == len(list(noise_dir.glob('*.npy'))), f'Missing adv. noises in {noise_dir}'
                for index in img_df.index:
                    npy_path = noise_dir / f"{img_df.loc[index,'name']}.npy"
                    n = np.load(npy_path)
                    #print(n.shape, 255*n.min(), 255*n.max(), n.dtype)
                    list_noise.append(n)
            rey = np.stack(list_noise,-1)
            rey = np.abs(np.fft.fftshift(np.fft.fftn(rey, axes=(0,1)), axes=(0,1)))**2
            rey = np.mean(rey,(-1,-2))
            energy2 = np.mean(rey)
            rey = rey / 4 / energy2
            plt.imsave(out_path,  rey.clip(0, 1), vmin=0, vmax=1)

def main(config_yaml: str, img_csv: str, adv_dir: str, out_dir: str):
    config = read_config(config_yaml)
    list_net = [net['name'] for net in config['networks']]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_df = pd.read_csv(img_csv)
    for atk in config['attacks']:
        atk_name = get_atk_dir_name(atk['name'], atk['args'])
        if atk['name'] == 'UNI':
            save_fft_uni(atk_name, list_net, adv_dir, out_dir)
        else:
            save_fft(atk_name, list_net, img_df, adv_dir, out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save residue fft.')

    parser.add_argument('--config', type=str, default='./configs/latent.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--img_csv', type=str, default='./AdversarialRobustnessCLIP/latent/list_images_latent.csv',
                        help='Path to the CSV file containing image list')
    parser.add_argument('--adv_dir', type=str, default='./images/latent',
                        help='Directory to save adversarial images')
    parser.add_argument('--out_dir', type=str, default='./fft_images',
                        help='Directory to save results')

    args = parser.parse_args()
    main(args.config, args.img_csv, args.adv_dir, args.out_dir)








