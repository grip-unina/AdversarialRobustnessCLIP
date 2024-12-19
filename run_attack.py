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

from attacks import select_attack_dict, attack_images, universal, robust_attack, pgd
import argparse
from pathlib import Path
from utils import read_config, get_atk_dir_name
from networks import load_network

def main(config: str, img_csv: str, dataset_dir: str, out_dir: str, device = 'cuda'):
    config = read_config(config)
    for net in config['networks']:
        namenet = net['name']
        for attack in config['attacks']:
            try:
                atk_name = attack['name']
                args = attack['args']
                atk_dir_name = get_atk_dir_name(atk_name, args)
                adv_img_dir = Path(out_dir, 'png', namenet, atk_dir_name)
                noise_dir = Path(out_dir, 'noise', namenet, atk_dir_name)

                print(f'Model name: {namenet} \nAttack name: {atk_name} \nArgs: {args}')

                if atk_name == 'UNI':
                    if config['attack_fake']:
                        print('Attacking fake images...')
                        universal(namenet, attack, config['fake_label'], img_csv, dataset_dir, adv_img_dir, noise_dir)
                    if config['attack_real']:
                        print('Attacking real images...')
                        universal(namenet, attack, config['real_label'], img_csv, dataset_dir, adv_img_dir, noise_dir)
                elif atk_name == 'RFGSM':
                    model = load_network(namenet).to(device).eval()
                    if config['attack_fake']:
                        print('Attacking fake images...')
                        robust_attack(model, attack, config['fake_label'], img_csv, dataset_dir, adv_img_dir, noise_dir)
                    if config['attack_real']:
                        print('Attacking real images...')
                        robust_attack(model, attack, config['real_label'], img_csv, dataset_dir, adv_img_dir, noise_dir)
                elif 'pgd' in atk_name:
                    model = load_network(namenet).to(device).eval()
                    if config['attack_fake']:
                        print('Attacking fake images...')
                        pgd(model, attack, config['fake_label'], img_csv, dataset_dir, adv_img_dir, noise_dir, device)
                    if config['attack_real']:
                        print('Attacking real images...')
                        pgd(model, attack, config['real_label'], img_csv, dataset_dir, adv_img_dir, noise_dir, device)
                else:
                    model = load_network(namenet).to(device)
                    atk = select_attack_dict(model, attack)
                    if config['attack_fake']:
                        print('Attacking fake images...')
                        attack_images(atk, config['fake_label'], img_csv, dataset_dir, adv_img_dir, noise_dir)
                    if config['attack_real']:
                        print('Attacking real images...')
                        attack_images(atk, config['real_label'], img_csv, dataset_dir, adv_img_dir, noise_dir)
                        
            except Exception as error:
                print("An exception occurred:", error)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="YAML config file", default='./configs/latent.yaml')
    parser.add_argument("--img_csv", type=str, help="Images csv", default='./AdversarialRobustnessCLIP/latent/list_images_latent.csv')
    parser.add_argument("--dataset_dir", type=str, help="Dataset directory", default='./AdversarialRobustnessCLIP')
    parser.add_argument("--out_dir", type=str, help="Output directory", default='./images/latent')
    parser.add_argument("--device", type=str, help="device", default='cuda')

    args = vars(parser.parse_args())
    main(**args)
