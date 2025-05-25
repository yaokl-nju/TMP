import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from .arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
# from .data import TrainDatasetForEmbedding, EmbedCollator
# from .modeling import BiEncoderModel
from .data_mtl import TrainDatasetForEmbedding, EmbedCollator
from .modeling_corda import BiEncoderModel
from .trainer import BiTrainer

import torch
import json

from .build_models.cov_func import calib_cov_distribution
from .build_models.build_model import build_model

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)
    torch.backends.cudnn.deterministic = True

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)

    model = BiEncoderModel(model_name=model_args.model_name_or_path,
                           normlized=training_args.normlized,
                           sentence_pooling_method=training_args.sentence_pooling_method,
                           negatives_cross_device=training_args.negatives_cross_device,
                           temperature=training_args.temperature,
                           use_inbatch_neg=training_args.use_inbatch_neg,
                           lora_r=training_args.lora_r,
                           lora_alpha=training_args.lora_alpha,
                           )

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    calib_cov_distribution(
        model,
        train_dataset,
        use_cache=True,
        calib_dataset=data_args.calib_dataset_name,
        calib_size=len(train_dataset) * data_args.real_batch_size,
        seed=123456
    )

    # Training
    # trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    # if trainer.is_world_process_zero():
    #     tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

# 修改
# 1. arguments.py修改：增加lora_r等参数
# 2. cov_func.py修改：传入trainer, dataloader;之后在本文件的trainer.train()之前调用conv_func()函数。
# 3. build_model.py修改：build_model()
# 4. modeling_corda.py修改：def save()
# 5. 非常重要的一点：sentence_transformer.models.Transformer.py中，self._load_model(..., trust_remote_code=True, ...)

# 住
# 1. 增加传入参数，修改arguments.py
# 2. 计算不同统计量，修改conv_func.py
# 3. 传入不同格式数据，修改modeling_corda.py forward()函数，data.py
# 4. 实现不同初始化方法，修改build_model.py
# 5. 非常重要的一点，sentence_transformer.models.models.Transformer.py中，self._load_model(..., trust_remote_code=True, ...)
# 6. 适配其它高效微调方法；
# 6.1 modeling_corda.py def save()
# 6.2 build_model.py def build_model()  
# 6.3 configurations_bert.py & modeling_bert.py