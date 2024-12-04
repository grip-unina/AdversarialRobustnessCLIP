import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
from functools import partial
from multiprocessing import Pool
from pathlib import Path

def save_image(img_path, pert, adv_dir):
    adv_path = adv_dir / Path(img_path).name
    img = np.float32(Image.open(img_path))/255
    adv_img = img+pert
    adv_img = np.clip(adv_img, 0, 1)
    Image.fromarray((adv_img * 255).astype(np.uint8)).save(adv_path)

def save_pert_images(pert, img_csv, dataset_dir, label, adv_dir):
    adv_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(img_csv)
    filenames = df[f'filename{label}'].apply(lambda x: Path(dataset_dir, x))
    assert len(filenames)>0, f'No images...'

    func = partial(save_image, pert=pert, adv_dir=adv_dir)
    n_proc = 200
    with Pool(n_proc) as p:
        p.map(func, filenames)
