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

from .resnet_attack import resnet50
from .clip_attack import ClipLinear, ClipE2E
from pathlib import Path

def load_weights(model, model_path):
    from torch import load
    dat = load(model_path, map_location='cpu')
    if 'model' in dat:
        if ('module._conv_stem.weight' in dat['model']) or \
           ('module.fc.fc1.weight' in dat['model']) or \
           ('module.fc.weight' in dat['model']):
            model.load_state_dict(
                {key[7:]: dat['model'][key] for key in dat['model']})
        else:
            model.load_state_dict(dat['model'])
    elif 'model_state_dict' in dat:
        from torch import cat
        model.load_state_dict({_[7:]: dat['model_state_dict'][_] if _[7:]!='conv1.weight' else cat((dat['model_state_dict'][_],dat['model_state_dict'][_],dat['model_state_dict'][_]),1)/3.0   for _ in  dat['model_state_dict']})
    elif 'state_dict' in dat:
        model.load_state_dict(dat['state_dict'])
    elif 'net' in dat:
        model.load_state_dict(dat['net'])
    elif 'main.0.weight' in dat:
        model.load_state_dict(dat)
    elif '_fc.weight' in dat:
        model.load_state_dict(dat)
    elif 'conv1.weight' in dat:
        model.load_state_dict(dat)
    else:
        print(list(dat.keys()))
        assert False
    return model

def load_network(namenet):
    weights_dir = './weights_AdversarialRobustnessCLIP'
    model_path = Path(weights_dir, f'{namenet}.pth')
    if namenet == 'GRAG_latent_r50':
        model = resnet50(num_classes=1, stride0=1)
        model = load_weights(model, model_path)
    elif namenet == 'CORV_latent_r50':
        model = resnet50(num_classes=1, stride0=1)
        model = load_weights(model, model_path)
    elif namenet == 'WANG_latent_r50':
        model = resnet50(num_classes=1, stride0=2)
        model = load_weights(model, model_path)
    elif namenet == 'OJHA_latent_clip':
        model = ClipLinear()
        model = load_weights(model, model_path)
    elif namenet == 'CORV_latent_clipe2e':
        model = ClipE2E()
        model = load_weights(model, model_path).float()
    elif namenet == 'GRAG_latent_clipe2e':
        model = ClipE2E()
        model = load_weights(model, model_path).float()
    elif namenet == 'CORV_latent_cn':
        from .timm_attack import WrapperTimm
        model = WrapperTimm('convnext_tiny.fb_in22k_ft_in1k', stride1=True, norm_type='resnet')
        model = load_weights(model, model_path)
        #GRIP_latent_cn  arch: timm_s1_convnext_tiny_in22ft1k norm_type: resnet
    elif namenet == 'OJHA_latent_blip':
        from .transformer_attack import TransformerLinear
        model = TransformerLinear('BLIP2_base',norm_type='clip')
        model = load_weights(model, model_path)
    else:
        assert False
    return model
