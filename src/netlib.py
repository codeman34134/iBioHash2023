import torch
import torch.nn as nn
import pretrainedmodels as ptm
import torch.nn.functional as F
import timm

from models import build_model

def initialize_weights(model):
    for idx,module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0,0.01)
            module.bias.data.zero_()

def rename_attr(model, attr, name):
    setattr(model, name, getattr(model, attr))
    delattr(model, attr)

# def networkselect(opt,config):
#     if opt.arch == 'resnet50':
#         network =  ResNet50(opt)
#     elif opt.arch == 'ViTB32':
#         network = ViTB32(opt)
#     elif opt.arch == 'ViTB16':
#         network = ViTB16(opt)
#     elif opt.arch == 'SwinL':
#         network = build_model(config,opt)
#         if opt.resume:
#             weights = torch.load(opt.resume,map_location='cpu')
#             del weights['model']['head.weight']
#             del weights['model']['head.bias']
#             if config.DATA.IMG_SIZE==768:
#                 del weights['model']['layers.0.blocks.1.attn_mask']
#                 del weights['model']['layers.1.blocks.1.attn_mask']
#                 del weights['model']['layers.3.blocks.0.attn.relative_coords_table']
#                 del weights['model']['layers.3.blocks.0.attn.relative_position_index']
#                 del weights['model']['layers.3.blocks.1.attn.relative_coords_table']
#                 del weights['model']['layers.3.blocks.1.attn.relative_position_index']
#             network.load_state_dict(weights['model'], strict=False)
#     else:
#         raise Exception('Network {} not available!'.format(opt.arch))
#     return network

def networkselect(opt,config):
    if opt.arch == 'resnet50':
        network =  ResNet50(opt)
    elif opt.arch == 'ViTB32':
        network = ViTB32(opt)
    elif opt.arch == 'ViTB16':
        network = ViTB16(opt)
    elif opt.arch == 'SwinL':
        network = build_model(config,opt)
        # if opt.resume:
        #     checkpoint = torch.load(opt.resume,map_location='cpu')
        #     state_dict = checkpoint['model']
        #
        #     # delete relative_position_index since we always re-init it
        #     relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        #     for k in relative_position_index_keys:
        #         del state_dict[k]
        #
        #     # delete relative_coords_table since we always re-init it
        #     relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
        #     for k in relative_position_index_keys:
        #         del state_dict[k]
        #
        #     # delete attn_mask since we always re-init it
        #     attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        #     for k in attn_mask_keys:
        #         del state_dict[k]
        #
        #     # bicubic interpolate relative_position_bias_table if not match
        #     relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        #     for k in relative_position_bias_table_keys:
        #         relative_position_bias_table_pretrained = state_dict[k]
        #         relative_position_bias_table_current = network.state_dict()[k]
        #         L1, nH1 = relative_position_bias_table_pretrained.size()
        #         L2, nH2 = relative_position_bias_table_current.size()
        #         if nH1 != nH2:
        #             pass
        #         else:
        #             if L1 != L2:
        #                 # bicubic interpolate relative_position_bias_table if not match
        #                 S1 = int(L1 ** 0.5)
        #                 S2 = int(L2 ** 0.5)
        #                 relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
        #                     relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
        #                     mode='bicubic')
        #                 state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
        #
        #     # bicubic interpolate absolute_pos_embed if not match
        #     absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
        #     for k in absolute_pos_embed_keys:
        #         # dpe
        #         absolute_pos_embed_pretrained = state_dict[k]
        #         absolute_pos_embed_current = network.state_dict()[k]
        #         _, L1, C1 = absolute_pos_embed_pretrained.size()
        #         _, L2, C2 = absolute_pos_embed_current.size()
        #         if C1 != C1:
        #             pass
        #         else:
        #             if L1 != L2:
        #                 S1 = int(L1 ** 0.5)
        #                 S2 = int(L2 ** 0.5)
        #                 absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
        #                 absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
        #                 absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
        #                     absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
        #                 absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3,
        #                                                                                                       1)
        #                 absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
        #                 state_dict[k] = absolute_pos_embed_pretrained_resized
        #
        #
        #     torch.nn.init.normal(network.head.bias, 0.,0.02)
        #     torch.nn.init.normal(network.head.weight, 0.,0.02)
        #     del state_dict['head.weight']
        #     del state_dict['head.bias']
        #
        #     msg = network.load_state_dict(state_dict, strict=False)
        #     print(msg)
        #
        #     del checkpoint
        #     torch.cuda.empty_cache()

    else:
        raise Exception('Network {} not available!'.format(opt.arch))



    return network

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class ViTB16(nn.Module):
    def __init__(self, opt, list_style=False, no_norm=False):
        super(ViTB16, self).__init__()
        self.pars = opt
        if not opt.not_pretrained:
            print('Getting pretrained weights...')
            self.model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True,img_size = 768)
        else:
            print('Not utilizing pretrained weights!')
            self.model = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.gem = GeM()
        self.model.head = torch.nn.Linear(self.model.head.in_features, opt.embed_dim)
        self.model.layer_norm = torch.nn.LayerNorm(self.model.head.in_features)

    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        x = self.model.pre_logits(x[:, 0])
        x = self.model.layer_norm(x)
        x = self.model.head(x)
        return torch.nn.functional.normalize(x, dim=-1)

class ViTB32(nn.Module):
    def __init__(self, opt, list_style=False, no_norm=False):
        super(ViTB32, self).__init__()
        self.pars = opt
        if not opt.not_pretrained:
            print('Getting pretrained weights...')
            self.model = timm.create_model('vit_base_patch32_224_in21k', pretrained=True)
        else:
            print('Not utilizing pretrained weights!')
            self.model = timm.create_model('vit_base_patch32_224', pretrained=False)
        self.gem = GeM()
        self.model.head = torch.nn.Linear(self.model.head.in_features, opt.embed_dim)
        self.model.layer_norm = torch.nn.LayerNorm(self.model.head.in_features)

    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        x = self.model.pre_logits(x[:, 0])
        x = self.model.layer_norm(x)
        x = self.model.head(x)
        return torch.nn.functional.normalize(x, dim=-1)

class ResNet50(nn.Module):
    def __init__(self, opt, list_style=False, no_norm=False):
        super(ResNet50, self).__init__()
        self.pars = opt
        if not opt.not_pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)
        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None
        self.gem = GeM()
        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)
        self.model.layer_norm = torch.nn.LayerNorm(self.model.last_linear.in_features)
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        #x = self.model.avgpool(x)
        x = self.gem(x)
        x = x.view(x.size(0),-1)
        x = self.model.layer_norm(x)
        mod_x = self.model.last_linear(x)
        return torch.nn.functional.normalize(mod_x, dim=-1)
