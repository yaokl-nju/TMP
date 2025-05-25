import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

@torch.no_grad()
def calib_cov_distribution(
    model,
    trainer,
    calib_loader,
    use_cache=True,
    calib_dataset="wiki",
    calib_size=256,
    seed=None
):
    target_modules = ["query", "value", "dense"]
    model_id = model.model.config._name_or_path
    cache_file = (
        f"cache/{model_id.replace('/', '_')}_covariance_matrices_from_{calib_dataset}_seed_{seed}.pt"
    )

    if os.path.exists(cache_file) and use_cache:
        print(f"covariance cache file found: {cache_file}...")
        all_covariance_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.model.named_modules():
            if isinstance(module, nn.Linear) and any([ti in name for ti in target_modules]):
                module.covariance_matrix = all_covariance_matrix[name].to(module.weight.device)
        return
    model.eval()

    print(f"building covariance file: {cache_file}")

    def hook(module, input, output):
        input = input[0].detach().squeeze(0).data  ## (2048, dim)

        # input = input.float()
        # if input.dim() > 2:
        #     input = input.mean(1)

        input = input.view(-1, input.size(-1))
        input = input.float()
        input = input / torch.max(input).abs()

        # covariance = input.t() @ input ## (dim, dim)
        if torch.isnan(input).any():
            print("nan detected")
            raise Exception("nan in input, break")
        if torch.isinf(input).any():
            print("inf detected")
            raise Exception("inf in input, break")

        covariance = input.t().matmul(input)

        if torch.isnan(covariance).any():
            print("nan detected")
            raise Exception("nan in covariance, break")
        if torch.isinf(covariance).any():
            print("inf detected")
            raise Exception("inf in covariance, break")
        
        module.covariance_matrix += covariance / calib_size
        # module.covariance_matrix = (module.covariance_matrix + covariance) / 2

        del covariance, input

    for name, module in model.model.named_modules():
        if isinstance(module, nn.Linear) and any([ti in name for ti in target_modules]):
            module.covariance_matrix = 0
            module.register_forward_hook(hook)

    calib_loader = trainer.get_train_dataloader()
    for batch in tqdm(calib_loader, desc="In build_models.cov_func.py: Calculate Covariance Matrix Begin ... "):
        batch = trainer._prepare_inputs(batch)
        # batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

    all_covariance_matrix = {}
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Linear) and any([ti in name for ti in target_modules]):
            module._forward_hooks.clear()
            if torch.isnan(module.covariance_matrix).any():
                print("nan detected")
                raise Exception("nan in covariance")
            if torch.isinf(module.covariance_matrix).any():
                print("inf detected")
                raise Exception("inf in covariance")
            # module.covariance_matrix = module.covariance_matrix / 256
            all_covariance_matrix[name] = module.covariance_matrix
    torch.save(all_covariance_matrix, cache_file)
    print("covariance matrices saved")