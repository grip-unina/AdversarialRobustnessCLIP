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
import timm
from torchvision.transforms import Normalize

class WrapperTimm(torch.nn.Module):
    def __init__(self, name_arch, stride1=True, norm_type='resnet'):
        super(WrapperTimm, self).__init__()
        self.model = timm.create_model(name_arch, num_classes=1, pretrained=True)
        
        if stride1:
            if 'conv1' in dir(self.model):
                self.model.conv1.stride=(1,1)
                if 'maxpool' in dir(self.model):
                    self.model.maxpool.stride=(1,1)
            elif '_conv_stem' in dir(self.model):
                self.model._conv_stem.stride = (1,1)
            elif 'stem' in dir(self.model):
                self.model.stem[0].stride = (1,1)
            else:
                assert False
        
        if norm_type=='resnet':
            self.prepros = Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        elif norm_type=='clip':
            self.prepros = Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))
        else:
            assert False
        
    def load_state_dict(self, state):
        self.model.load_state_dict(state)

    def forward(self, x):
        y = self.model(self.prepros(x))
        return torch.cat((torch.zeros_like(y),y),1)
    