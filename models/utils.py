import numpy as np
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from .robust_vit import VisionTransformer
from .wrapper import ModelWrapper
from .noise_resnet import noise_resnet20
from .vanilla_resnet import vanilla_resnet20

class RandomDefense(nn.Module):
    def __init__(self, noise, scale=None) -> None:
        super().__init__()
        self.noise = noise
        if scale is not None:
            self.scale = nn.Parameter(scale)
        else:
            self.scale = None

    def forward(self, x):
        if self.scale is not None:
            out = x + self.noise * torch.randn_like(x) * self.scale
        else:
            out = x + self.noise * torch.randn_like(x)
        return out

def defense_token(x, defense_type, noise_sigma, scale=None):
    if defense_type == 'gauss_filter':
        gauss_x = torch.tensor(gauss_x).cuda()
        return gauss_x
    elif defense_type == 'random_noise':
        noise = torch.randn_like(x) * noise_sigma
        if scale is not None:
            noise = noise * scale.to(x)
        return x + noise    
    elif defense_type == 'laplace':
        d = torch.distributions.laplace.Laplace(torch.zeros_like(x), noise_sigma * torch.ones_like(x))
        return x + d.sample()
    elif defense_type == 'identical':
        return x

def add_defense(model_name, model, defense_type, def_position, noise, layer_index=-1, scale=False, dset_name=None):
    if isinstance(layer_index, int):
        layer_index = [layer_index]
    print(model_name)
    if 'resnetv2' in model_name:
        if def_position == 'hidden_feature':
            print(layer_index, noise)
            model.stages = nn.Sequential(*sum([[b, RandomDefense(noise, None)] 
                                               if i in layer_index or -1 in layer_index else [b] 
                                               for i, b in enumerate(model.stages)], []))
    elif ('resnet' in model_name or 'res50' in model_name) and 'vanilla' not in model_name:
        # print('vanilla' not in model_name)
        if def_position == 'hidden_feature':
            def forward_features_new(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.act1(x)
                x = self.maxpool(x)

                if 0 in layer_index or -1 in layer_index:
                    x = defense_token(x, defense_type, noise)
                
                x = self.layer1(x)
                if 1 in layer_index or -1 in layer_index:
                    x = defense_token(x, defense_type, noise)
                
                x = self.layer2(x)
                if 2 in layer_index or -1 in layer_index:
                    x = defense_token(x, defense_type, noise)
                
                x = self.layer3(x)
                if 3 in layer_index or -1 in layer_index:
                    x = defense_token(x, defense_type, noise, scale=std if scale else None)
                
                x = self.layer4(x)
                return x

            model.forward_features = types.MethodType(forward_features_new, model)
    elif 'vgg' in model_name:
        if def_position == 'hidden_feature':
            
            def forward_features_new(self, x):
                c = 0
                for l in self.features: 
                    if 'conv' in l._get_name().lower():
                        if c in layer_index or layer_index == [-1]:
                            x = defense_token(x, defense_type, noise)
                        c += 1
                    x = l(x)
                return x
            model.forward_features = types.MethodType(forward_features_new, model)
    elif 'vit' in model_name:
        if def_position == 'hidden_feature':
            std = torch.load(f'stats/{model_name}_cifar10_all_std.pth')
            model.blocks = nn.Sequential(*sum([[RandomDefense(noise, std[i] if scale else None), b] if i in layer_index or -1 in layer_index else [b] for i, b in enumerate(model.blocks)], []))
    elif 'deit' in model_name:
        if def_position == 'hidden_feature':
            std = torch.load(f'stats/{model_name}_cifar10_all_std.pth')
            model.blocks = nn.Sequential(*sum([[RandomDefense(noise, std[i] if scale else None), b] if i in layer_index or -1 in layer_index else [b] for i, b in enumerate(model.blocks)], []))
    elif 'mixer' in model_name or 'resmlp' in model_name:
        if def_position == 'hidden_feature':
            model.blocks = nn.Sequential(*sum([[RandomDefense(noise), b] if i in layer_index or -1 in layer_index else [b] for i, b in enumerate(model.blocks)], []))
    elif 'poolformer' in model_name:
        if def_position == 'hidden_feature':
            for i in range(len(model.network)):
                if type(model.network[i]).__name__ == 'Sequential':
                    model.network[i] = nn.Sequential(*sum([[RandomDefense(noise), b] if type(b).__name__ == 'PoolFormerBlock' else [b] for b in model.network[i] ], []))
    elif 'vanilla_resnet' in model_name:
        if def_position == 'hidden_feature':
            def forward_new(self, x):
                x = self.conv_1_3x3(x)
                x = F.relu(self.bn_1(x), inplace=True)
                x = self.stage_1(x)
                x = self.stage_2(x)
                x = self.stage_3(x)
                x = defense_token(x, defense_type, noise)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
            model.forward = types.MethodType(forward_new, model)
    return model

def create_robust_model(model_name, dataset, n_cls, noise, defense, def_position, device='cpu', layer_index=-1, blackbox=True, scale=False):
    if 'noise_resnet' in model_name:
        assert dataset == 'cifar10'
        base_model = noise_resnet20()
    elif 'vanilla_resnet' in model_name:
        assert dataset == 'cifar10'
        base_model = vanilla_resnet20()
    elif model_name == 'advres50_gelu':
        base_model = timm.create_model('resnet50')
        base_model.load_state_dict(torch.load('pretrain/advres50_gelu.pth')['model'])

    elif 'vit' not in model_name and 'torchvision' not in model_name:
        base_model = timm.create_model(model_name, pretrained=False)
    elif 'vit' in model_name:
        if 'small' in model_name:
            kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
        else:
            kwargs = dict()
        base_model = VisionTransformer(weight_init='skip', num_classes=1000, defense_cls=defense, noise_sigma=noise, def_position=def_position, **kwargs)
        base_model.default_cfg = timm.create_model(model_name).default_cfg
        base_model.layer_index = layer_index
        base_model.set_defense(True)

    if dataset == 'cifar10':
        # breakpoint()
        model_path = model_name.replace('.', '_')
        state_dict = torch.load(f'pretrain/{model_path}_{dataset}.pth.tar', map_location='cpu')['state_dict']
        if 'noise_resnet' in model_name or model_name == 'vanilla_resnet_adv':
            state_dict = {k[2:]:v for (k, v) in state_dict.items() if k[0] != '0'}
        base_model.load_state_dict(state_dict)
    elif (dataset == 'imagenet' or dataset == 'imagenet_baseline') and 'torchvision' not in model_name and 'adv' not in model_name:
        base_model.load_state_dict(timm.create_model(model_name, pretrained=True).state_dict())
            
    if 'noise_resnet' in model_name or 'vanilla_resnet' in model_name:
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif 'torchvision' in model_name:
        import torchvision
        base_model = torchvision.models.resnet50(pretrained=True)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif 'adv' in model_name:
        mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    else:
        mean = base_model.default_cfg['mean']
        std = base_model.default_cfg['std']
    base_model.noise_sigma = noise
    if 'torchvision' not in model_name:
        base_model = add_defense(model_name, base_model, defense, def_position, noise, layer_index, scale=scale, dset_name=dataset)
    if blackbox:
        model = ModelWrapper(base_model, num_classes=n_cls, device=device, def_position=def_position, mean=mean, std=std)
        return model
    else:
        return base_model, mean, std