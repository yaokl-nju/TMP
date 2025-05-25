import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from .arguments import DataArguments, RetrieverTrainingArguments
import time
import copy
import re
import numpy as np
import os

from torch.utils.data import RandomSampler
from sentence_transformers import SentenceTransformer

@dataclass
class TaskBatchIndex:
    name: str
    batch_index: list[int]

Categories = ['classification', 'clustering', 'pairclassification', 'reranking', 'retrieval', 'STS']
NumEvalDataSets = {'classification': 9, 'clustering': 4, 'pairclassification': 2, 'reranking': 4, 'retrieval': 8, 'STS': 8}

class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        start_time = time.time()
        self.dataset = {}

        for cat in args.task_name.split(' '):
            train_datasets_sub = []
            dirlist = [os.path.join(args.train_data, cat)]
            while len(dirlist) > 0:
                curdir = dirlist.pop(0)
                for dir in os.listdir(curdir):
                    if os.path.isdir(os.path.join(curdir, dir)):
                        dirlist.append(os.path.join(curdir, dir))
                    else:
                        temp_dataset = datasets.load_dataset('json', data_files=os.path.join(curdir, dir), split='train')
                        train_datasets_sub.append(temp_dataset)
            self.dataset[cat] = datasets.concatenate_datasets(train_datasets_sub)

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = sum([len(self.dataset[cat]) for cat in self.dataset.keys()])
        print("Time cost for loading large-scale datasets: {:1.f}".format(time.time() - start_time))
        print("Total number of examples: ", self.total_len)

        self.refresh_batch_data()

    def __len__(self):
        # return self.total_len
        return len(self.task_batch_index_list)
    
    def refresh_batch_data(self):
        self.task_batch_index_list: list[TaskBatchIndex] = []
        for cat in self.dataset.keys():
            batch_size = self.args.real_batch_size
            num_samples = (len(self.dataset[cat]) // batch_size) * batch_size
            buffer = []
            for i in RandomSampler(self.dataset[cat], num_samples=num_samples):
                buffer.append(i)
                if len(buffer) == batch_size:
                    self.task_batch_index_list.append(TaskBatchIndex(name=cat, batch_index=buffer))
                    buffer = []
        self.random_index_list = list(RandomSampler(self.task_batch_index_list))

    def get_batch_spo_data(self, batch_data, task_name):
        # reranking, retrieval, STS
        query = [data['query'] for data in batch_data]
        passages = []
        for data in batch_data:
            passages.append(random.choice(data['pos']))
            neg_raw = data['neg']
            if len(neg_raw) < self.args.train_group_size - 1:
                num = math.ceil((self.args.train_group_size - 1) / len(neg_raw))
                negs = random.sample(neg_raw * num, self.args.train_group_size - 1)
            elif len(neg_raw) == self.args.train_group_size - 1:
                negs = neg_raw
            else:
                negs = random.sample(neg_raw, self.args.train_group_size - 1)
            passages.extend(negs)

        return query, passages, True, self.args.train_group_size
    
    def get_batch_other_data(self, batch_data, task_name):
        # reranking, retrieval, STS
        query = [data['query'] for data in batch_data]
        passages = []
        for data in batch_data:
            passages.append(random.choice(data['query']))
            
        return query, passages, False, self.args.train_group_size

    def get_batch_cls_data(self, batch_data, task_name):
        # clustering, classification
        groupsize = self.args.train_group_size
        query = [data['query'] for data in batch_data]
        passages = []
        for data in batch_data:
            neg_raw = data['neg_candidate'] if 'neg_candidate' in data and data['neg_candidate'] else data['neg']
            pos_raw = data['pos_candidate'] if 'pos_candidate' in data and data['pos_candidate'] else data['pos']

            passages.append(random.choice(pos_raw))
            if len(neg_raw) < groupsize - 1:
                num = math.ceil((groupsize - 1) / len(neg_raw))
                negs = random.sample(neg_raw * num, groupsize - 1)
            elif len(neg_raw) == groupsize - 1:
                negs = neg_raw
            else:
                negs = random.sample(neg_raw, groupsize - 1)
            passages.extend(negs)

        return query, passages, False, groupsize
    
    def get_batch_paircls_data(self, batch_data, task_name):
        # clustering, classification
        groupsize = 3

        query = [data['query'] for data in batch_data]
        passages = []
        for data in batch_data:
            passages.append(data['pos'][0])
            passages.extend(data['neg'])

        return query, passages, False, groupsize
    
    def __getitem__(self, item) -> Tuple[str, List[str]]:
        index = self.random_index_list[index]
        task_batch_index = self.task_batch_index_list[index]
        task_name = task_batch_index.name
        batch_index = task_batch_index.batch_index

        target_dataset = self.datset[task_name]
        batch_data = [target_dataset[i] for i in batch_index]
        
        if task_name in ['reranking', 'retrieval', 'STS']:
            query, passages, in_batch_neg, group_size = self.get_batch_spo_data(batch_data, task_name)
        elif task_name in ['clustering', 'classification']:
            query, passages, in_batch_neg, group_size = self.get_batch_cls_data(batch_data, task_name)
        elif task_name in ['pairclassification']:
            query, passages, in_batch_neg, group_size = self.get_batch_paircls_data(batch_data, task_name)
        elif task_name == 'others':
            query, passages, in_batch_neg, group_size = self.get_batch_other_data(batch_data, task_name)
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        
        return query, passages, in_batch_neg, task_name


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 128
    passage_max_len: int = 512

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        query = features[0][0]
        passage = features[0][1]
        in_batch_neg = features[0][2]
        task_name = features[0][3]

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated, "in_batch_neg": in_batch_neg, "task_name": task_name}
