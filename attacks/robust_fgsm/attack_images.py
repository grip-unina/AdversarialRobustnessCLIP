from .robust_fgsm import robust_fgsm
import pandas as pd
from pathlib import Path
from torchvision.transforms import ToTensor, ToPILImage
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

def robust_attack(input_img, label, model, model_type, my_atk: dict, cuda = True):
    assert my_atk['name'] != 'FGSM', 'Wrong attack'
    
    args = my_atk['args'].copy()
    if 'eps' in args:
        args['eps'] = args['eps']/255.0
    if 'alpha' in args:
        args['alpha'] = args['alpha']/255.0

    return robust_fgsm(input_img, label, model, model_type, cuda, **args)

def attack_images(model, atk_dict, label, img_csv, dataset_dir, adv_img_dir, noise_dir):
    pil2torch = ToTensor()
    torch2pil = ToPILImage()
    model_type = None
    
    df = pd.read_csv(img_csv)
    img_paths = df[f'filename{label}'].apply(lambda x: Path(dataset_dir, x))
            
    adv_img_dir = Path(adv_img_dir, f'adv{label}')
    adv_img_dir.mkdir(parents=True, exist_ok=True)
    noise_dir = Path(noise_dir, f'adv{label}')
    noise_dir.mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(img_paths):
        adv_path = adv_img_dir / img_path.name
        if not adv_path.exists():
            img = pil2torch(Image.open(img_path))
            processed_image = img[None, :]
            perturbed_image, attack_meta_data = robust_attack(processed_image, label, model, model_type, atk_dict)
            torch2pil(perturbed_image.squeeze()).save(adv_path)

            noise = attack_meta_data['noise'].squeeze().detach().numpy()
            noise = np.transpose(noise, (1, 2, 0))
            np.save(noise_dir / adv_path.with_suffix('.npy').name, noise)