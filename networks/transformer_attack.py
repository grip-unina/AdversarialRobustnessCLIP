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

import torch
import torch.nn as nn
from .resnet_attack import ChannelLinear
import transformers
from torchvision.transforms import Normalize

list_transformers = {
'FLAVA'       : {'name': "facebook/flava-full"          , 'class': 'FlavaImageModel', 'method': '__call__'            , 'norm':'clip', 'num_features':  768},
'CLIP_ViTb_32': {'name': "openai/clip-vit-base-patch32" , 'class': 'CLIPVisionModel', 'method': '__call__'            , 'norm':'clip', 'num_features':  768},
'CLIP_ViTl_14': {'name': "openai/clip-vit-large-patch14", 'class': 'CLIPVisionModel', 'method': '__call__'            , 'norm':'clip', 'num_features': 1024},
'BLIP2_base'  : {'name': "Salesforce/blip2-opt-2.7b"    , 'class': 'Blip2Model'     , 'method': 'get_image_features'  , 'norm':'clip', 'num_features': 1408},
'BLIP2_baseqf': {'name': "Salesforce/blip2-opt-2.7b"    , 'class': 'Blip2Model'     , 'method': 'get_qformer_features', 'norm':'clip', 'num_features':  768},
}


def get_transformer_backboone(name):
    dat = list_transformers[name]
    model = getattr(transformers, dat['class']).from_pretrained(dat['name'])
    method = dat['method']
    num_features = dat['num_features']
    return model, method, num_features


class TransformerLinear(nn.Module):
    def __init__(self, transformer_name, norm_type='clip'):
        super(TransformerLinear, self).__init__()
        backbone, self.method, self.num_features = get_transformer_backboone(transformer_name)
        self.bb = [backbone, ]
        self.fc = ChannelLinear(self.num_features, 1)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)
        if norm_type=='resnet':
            self.prepros = Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        elif norm_type=='clip':
            self.prepros = Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))
        else:
            assert False

    def to(self, *args, **kwargs):
        self.bb[0].to(*args, **kwargs)
        super(TransformerLinear, self).to(*args, **kwargs)
        return self

    def forward_features(self, x):
        self.bb[0].eval()
        embedding = getattr(self.bb[0], self.method)(self.prepros(x))['pooler_output']
        return embedding
    
    def forward_head(self, x):
        y = self.fc(x)
        return torch.cat((torch.zeros_like(y),y),1)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))
