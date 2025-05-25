import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

class CorDA_adapter(nn.Module):
    def __init__(self, adapter_U, adapter_S, adapter_V, weight_residual, bias=None, sigma_fuse='UV', scaling=1.) -> None:
        super().__init__()
        U, S, V = adapter_U, adapter_S, adapter_V    # U: (m, r), V: (n, r), n==in_size, m==out_size
        rank = V.size(1)

        # corda
        self.weight = nn.Parameter(torch.empty((U.size(0), V.size(0))).to(adapter_U.device))
        self.weight.data = weight_residual
        self.weight.requires_grad = False

        if bias is not None:
            self.bias = nn.Parameter(torch.empty(U.size(0)).to(adapter_U.device))
            self.bias.data = bias
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)
        
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=False)
        self.BLinear = nn.Linear(V.size(0), V.size(1), bias=False)
        self.BLinear.weight.requires_grad = False

        if sigma_fuse == 'UV':
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == 'U':
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == 'V':
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()

        ## inflora
        # self.ALinear.weight.data = nn.Linear(V.size(1), weight_residual.size(0), bias=False)  ## r -> m
        # self.BLinear.weight.data = nn.Linear(V.size(0), V.size(1), bias=False)  ## n -> r
        # self.BLinear.weight.requires_grad = False
        ## self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()
        # self.BLinear.weight.data = V.t().contiguous()
        # nn.init.zeros_(self.ALinear.weight)

        self.scaling = scaling

    def forward(self, input):
        y = self.BLinear(input)
        y = self.scaling * self.ALinear(y) + F.linear(input, self.weight) + (self.bias if self.bias is not None else 0)

        # y = F.linear(input, self.weight) + (self.bias if self.bias is not None else 0)

        return y
    
    def decompose_to_adapter_cov(
            linear: nn.Linear,
            cov_aware=False,
            alpha=1,
            sigma_fuse='UV',
            r=16,
            lora_alpha=64
    ):
        rank = min(linear.in_features, linear.out_features)
        pretrained_w = linear.weight.data.float()#.cpu()
        if True:
            assert hasattr(linear, "covariance_marix")
            covariance_matrix = linear.covariance_marix.float()#.cpu()
        try:
            if True:
                U, S, V = torch.linalg.svd(covariance_matrix, full_matrices=False)
                V = V.transpose(0, 1)
        except:
            raise Exception(f"svd failed for {linear}")
        
        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        
        # nan or inf check
        # if (S!=S).any():
        if torch.isnan(S).any() or torch.isinf(S).any():
            raise Exception(f"nan or inf in S")
        # if (U!=U).any():
        if torch.isnan(U).any() or torch.isinf(U).any():
            raise Exception(f"nan or inf in U")
        # if (V!=V).any():
        if torch.isnan(V).any() or torch.isinf(V).any():
            raise Exception(f"nan or inf in V")
        
        U = U[:, -r:]   ## m, r
        S = S[-r:]      ## r
        V = V[:, -r:]   ## n, r

        scaling = lora_alpha / math.sqrt(r)
        S /= scaling

        # weight_residual = pretrained_w - scaling * (U @ torch.diag(S) @ V.transpose(0, 1))  ## m, n
        # # weight_residual = pretrained_w - U @ torch.diag(S) @ V.transpose(0, 1)  ## m, n

        # if torch.isnan(weight_residual).any() or torch.isinf(weight_residual).any():
        #     raise Exception(f"nan or inf in weight_residual")

        linear_with_adapter = CorDA_adapter(U, S, V, pretrained_w, bias, sigma_fuse, scaling)
        linear_with_adapter.to(linear.weight.dtype)#.cpu()
        linear_with_adapter.to(linear.weight.device)
        linear_with_adapter.weight = linear_with_adapter.weitght.to(linear.weight.dtype)

        # del pretrained_w, U, S, V, weight_residual, linear
        del pretrained_w, U, S, V, linear
        torch.cuda.empty_cache()
        
        return linear_with_adapter
    
    #@staticmethod
    def decompose_to_adapter(
            linear: nn.Linear,
            cov_aware=False,
            alpha=1,
            sigma_fuse='UV',
            r=16,
            lora_alpha=64
    ):
        rank = min(linear.in_features, linear.out_features)
        pretrained_w = linear.weight.data.float()#.cpu()
        if True:
            assert hasattr(linear, "covariance_marix")
            covariance_matrix = linear.covariance_marix.float()#.cpu()
            damp = 0.01
            while True:
                compensate = torch.diag(torch.ones(covariance_matrix.size(0)).to(covariance_matrix.device) * torch.mean(torch.diag(covariance_matrix)) * damp)
                fix_covariance_matrix = covariance_matrix + compensate
                cov_inv = torch.linalg.inv(fix_covariance_matrix)
                inv_error = torch.dist(fix_covariance_matrix @ cov_inv, torch.eye(covariance_matrix.size(0)).to(covariance_matrix.device))
                if inv_error.data < 0.05:
                    break
                else:
                    damp = damp * 2
            w = pretrained_w @ fix_covariance_matrix  ## w: out_dim, in_dim; covariance_matrix: in_dim, in_dim
        try:
            if True:
                # U, S, V = torch.svd_lowrank(w, q=rank)
                U, S, V = torch.linalg.svd(w, full_matrices=False)
                V = V.transpose(0, 1)
        except:
            raise Exception(f"svd failed for {linear}")
        

        if True:
            V = (V.t() @ cov_inv).transpose(0, 1)

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        
        # nan or inf check
        # if (S!=S).any():
        if torch.isnan(S).any() or torch.isinf(S).any():
            raise Exception(f"nan or inf in S")
        # if (U!=U).any():
        if torch.isnan(U).any() or torch.isinf(U).any():
            raise Exception(f"nan or inf in U")
        # if (V!=V).any():
        if torch.isnan(V).any() or torch.isinf(V).any():
            raise Exception(f"nan or inf in V")
        
        U = U[:, -r:]   ## m, r
        S = S[-r:]      ## r
        V = V[:, -r:]   ## n, r

        scaling = lora_alpha / r
        S /= scaling

        weight_residual = pretrained_w - scaling * (U @ torch.diag(S) @ V.transpose(0, 1))  ## m, n
        # weight_residual = pretrained_w - U @ torch.diag(S) @ V.transpose(0, 1)  ## m, n

        if torch.isnan(weight_residual).any() or torch.isinf(weight_residual).any():
            raise Exception(f"nan or inf in weight_residual")

        linear_with_adapter = CorDA_adapter(U, S, V, weight_residual, bias, sigma_fuse, scaling)
        linear_with_adapter.to(linear.weight.dtype)#.cpu()
        linear_with_adapter.to(linear.weight.device)
        linear_with_adapter.weight = linear_with_adapter.weitght.to(linear.weight.dtype)

        # del pretrained_w, U, S, V, weight_residual, linear
        del pretrained_w, U, S, V, linear, weight_residual
        torch.cuda.empty_cache()
        
        return linear_with_adapter
    
    def build_model(model, lora_r=256, lora_alpha=64):
        target_modules = ["query", "value", "dense"]

        module_dict = {name: module for name, module in model.model.named_modules()}
        full_name_dict = {module: name for name, module in model.model.named_modules()}
        linear_info = {}
        modules = [model.model]
        while len(modules) > 0:
            submodule = modules.pop()
            for name, raw_linear in submodule.named_children():
                if isinstance(raw_linear, nn.Linear) and any([ti in full_name_dict[raw_linear] for ti in target_modules]):
                    full_name = full_name_dict[raw_linear]
                    linear_info[raw_linear] = {
                        "father": submodule,
                        "name": name,
                        "full_name": full_name,
                    }
                else:
                    modules.append(raw_linear)
        my_layers_keys = []
        for name, module in model.model.named_modules():
            if isinstance(module, nn.Linear) and any([ti in name for ti in target_modules]):
                my_layers_keys.append(name)

        print('In build_model.py: ---- model before svd ----')

        for layername in tqdm(my_layers_keys):
            raw_linear = module_dict[layername]
            info = linear_info[raw_linear]
            with torch.no_grad():
                if True:
                    # if "lm_head" is layer_name:
                    #     continue
                    linear_with_adapter = decompose_to_adapter(
                        raw_linear,
                        cov_aware=True,
                        r=lora_r,
                        lora_alpha=lora_alpha
                    )
                    delattr(info['fathter'], info['name'])
                    delattr(raw_linear, "covarance_matrix")
                    setattr(info['father'], info['name'], linear_with_adapter)
                    del module_dict[layername], linear_info[raw_linear]
                    del raw_linear, info
                    torch.cuda.empty_cache()
        print('In build_model.py: ---- model after svd ----')