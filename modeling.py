import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel, BertModel
from transformers.file_utils import ModelOutput
import os
from sentence_transformers import models

logger = logging.getLogger(__name__)

def get_nb_trainable_parameters(model) -> tuple[int, int]:
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, 'ds_numel'):
            num_params = param.ds_numel
        
        if param.__class__.__name__ == 'Params4bit':
            num_bytes = param.quant_storage.itemsize if hasattr(param, 'quant_storage') else 1
            num_params = num_params * 2 * num_bytes
        
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True,
                 use_mrl: bool = False,
                 ):
        super().__init__()

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        if 'init' in model_name:
            for n, p self.model.named_parameters():
                if 'ALinear' not in n:
                    p.requires_grad = False
        for n, p self.model.named_parameters():
            print(n, p.requires_grad)
        
        trainable_params, all_param = get_nb_trainable_parameters(self.model)
        print(f"trainable params: {trainable_params}, all params: {all_param}")


        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config
        self.use_mrl = use_mrl

        self.mrl_dims = [128, 256, 512, 1024]
        self.fc = None
        self.soft_plus = nn.Softplus()

        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")
        if normlized:
            if self.temperature > 0.5:
                raise ValueError("Temperature should be smaller than 1.0 when use cosine similarity (i.e., normlized=True). Recommend to set it 0.01-0.1")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        # if self.normlized:
        #     p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, in_batch_neg: bool = None, task_name: str = None, teacher_score: Tensor = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            loss = torch.tensor(0.0, device=q_reps.device)

            for dim in self.mrl_dims:
                q_reps_sub, p_reps_sub = q_reps[:, :dim], p_reps[:, :dim]
                if self.normlized:
                    q_reps_sub = torch.nn.functional.normalize(q_reps_sub, dim=-1)
                    p_reps_sub = torch.nn.functional.normalize(p_reps_sub, dim=-1)

                if in_batch_neg:
                    scores = self.compute_similarity(q_reps_sub, p_reps_sub) / self.temperature # B B*G
                    scores = scores.view(q_reps.size(0), -1)

                    target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                    target = target * group_size
                    loss_i = self.compute_loss(scores, target)
                else:
                    scores = self.compute_similarity(q_reps_sub[:, None, :,], p_reps_sub.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

                    scores = scores.view(q_reps.size(0), -1)
                    if task_name != 'pairclassification':
                        target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                        loss_i = self.compute_loss(scores, target)
                    else:
                        scores_diff_p_hn = -(scores[:, 0].unsqueeze(0) - scores[:, 1:].reshape(-1, 1)).view(-1)
                        scores_diff_h_n = -0.02 * (scores[:, 1].unsqueeze(0) - scores[:, 2].unsqueeze(1)).view(-1)
                        loss_i = self.soft_plus(torch.logsumexp(torch.cat((scores_diff_p_hn, scores_diff_h_n)), dim=0))
                        # target1 = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                        # target2 = torch.ones(scores.size(0), device=scores.device, dtype=torch.long)
                        # loss_i1 = self.compute_loss(scores, target1)
                        # loss_i2 = self.compute_loss(scores, target2)
                        # loss_i = loss_i1 + loss_i2
                loss += loss_i
            loss /= len(self.mrl_dims)

        else:
            if self.normlized:
                q_reps = torch.nn.functional.normalize(q_reps, dim=-1)
                p_reps = torch.nn.functional.normalize(p_reps, dim=-1)

            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

        os.system("cp /opt/nas/n/.../mapping/configurations_bert.py " + output_dir)
        os.system("cp /opt/nas/n/.../mapping/modeling_bert.py " + output_dir)
