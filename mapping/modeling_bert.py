from transformers import BertModel
from .configurations_bert import CorDAbertConfig
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class CorDALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, scaling, bias=True):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=False)
        self.BLinear.weight.requires_grad = False
        # InfLoRA
        # nn.init.zeros_(self.ALinear.weight)

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight.requires_grad = False
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)
        self.scaling = scaling

    def forward(self, input):
        y = self.BLinear(input)
        y = self.scaling * self.ALinear(y) + F.linear(input, self.weight) + (self.bias if self.bias is not None else 0)
        return y
    
class CorDAForBert(BertModel):
    config_class = CorDAbertConfig
    def __init__(self, config: CorDAbertConfig):
        super().__init__(config)
        self.target_modules = ["query", "value", "dense"]

        self.lora_r = config.lora_r
        # self.scaling = config.lora_alpha / math.sqrt(config.lora_r)
        self.scaling = config.lora_alpha / config.lora_r

        full_name_dict = {module: name for name, module in self.named_modules()}
        linear_info = {}
        modules = [self]
        while len(modules) > 0:
            submodule = modules.pop()
            for name, raw_linear in submodule.named_children():
                if isinstance(raw_linear, nn.Linear) and any([ti in full_name_dict[raw_linear] for ti in self.target_modules]):
                    full_name = full_name_dict[raw_linear]
                    linear_info[raw_linear] = {
                        "father": submodule,
                        "name": name,
                        "full_name": full_name
                    }
                else:
                    modules.append(raw_linear)
                
            for name, module in self.named_modules():
                # if "lm_head" not in name and isinstance(module, nn.Linear) and any([ti in name for ti in self.target_modules]):
                if isinstance(module, nn.Linear) and any([ti in name for ti in self.target_modules]):
                    info=linear_info[module]
                    new_layer=CorDALinear(module.in_features, module.out_features, self.lora_r, self.scaling, bias=module.bias is not None)
                    setattr(info['father'], info['name'], new_layer)