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
import torch.nn.functional as F
import clip
from torchvision.transforms import Normalize
from .resnet_attack import ChannelLinear

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output
    
class ClipLinear(nn.Module):
    def __init__(self, num_classes=1, clip_name="ViT-L/14"):
        super(ClipLinear, self).__init__()
        self.prepros = Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        backbone, self.preprocess = clip.load(clip_name, jit=False)
        backbone = backbone.float().eval().visual
        self.num_features = backbone.output_dim
        self.bb = [backbone,]
        self.fc = ChannelLinear(self.num_features, num_classes)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

    def to(self, *args, **kwargs):
        self.bb[0].to(*args, **kwargs)
        super(ClipLinear, self).to(*args, **kwargs)
        return self
        
    def forward_features(self, x):
        self.bb[0].eval()
        embedding = self.bb[0](self.prepros(x))
        return embedding.flatten(1)
    
    def forward_head(self, x):
        y = self.fc(x)
        return torch.cat((torch.zeros_like(y),y),1)
        
    def forward(self, x):
        return self.forward_head(self.forward_features(x))

    
class feature_generator_clip(nn.Module):
    def __init__(self):
        super(feature_generator_clip, self).__init__()
        vit, _ = clip.load("ViT-B/16", device='cuda')
        vit = vit.visual
        vit.proj = None
        self.vit = vit

    def forward(self, input):
        self.vit.proj = None
        feat = self.vit.forward(input)
        return feat

class feature_embedder(nn.Module):

    def __init__(self):
        super(feature_embedder, self).__init__()
        self.bottleneck_layer_fc = nn.Linear(768, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_layer_fc, nn.ReLU(),
                                              nn.Dropout(0.5))

    def forward(self, input, norm_flag=True):
        feature = self.bottleneck_layer(input)
        if (norm_flag):
            feature_norm = torch.norm(
              feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)**0.5 * (2)**0.5
            feature = torch.div(feature, feature_norm)
        return feature

class classifier(nn.Module):

    def __init__(self):
        super(classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag=True):
        if (norm_flag):
            self.classifier_layer.weight.data = l2_norm(
              self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out

class ClipE2E(nn.Module):

    def __init__(self, norm_flag=True):
        super(ClipE2E, self).__init__()
        self.prepros = Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        self.backbone = feature_generator_clip()
        self.embedder = feature_embedder()
        self.classifier = classifier()
        self.norm_flag = norm_flag
    
    def load_state_dict(self, state):
        if 'backbone.vit.proj' in state:
            del state['backbone.vit.proj']
        super(ClipE2E, self).load_state_dict(state)

    def forward_features(self, x):
        x = self.backbone(self.prepros(x))
        x = self.embedder(x, self.norm_flag)
        return x
        
    def forward_head(self, x):
        classifier_out = self.classifier(x, self.norm_flag)
        y = classifier_out[:,0:1] - classifier_out[:,1:2]
        return torch.cat((torch.zeros_like(y),y),1)
        
    def forward(self, x):
        return self.forward_head(self.forward_features(x))
