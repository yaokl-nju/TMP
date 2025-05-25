from transformers import BertConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class CorDAbertConfig(BertConfig):
    model_type = "bert"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        lora_r=128,
        lora_alpha=64,
        **kwargs     
    ):
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        super().__init__(**kwargs)