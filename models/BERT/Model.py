# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import shutil

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from .utils import convert_examples_to_features
from .modeling import BertForSequenceClassification

class BERT:
    def __init__(self, args, num_train_examples):
        self.max_seq_length = args.max_sent_length
        self.train_batch_size = 32
        self.eval_batch_size = 8
        self.do_lower_case = True
        self.learning_rate = 2e-5
        self.warmup_proportion = 0.05
        self.gradient_accumulation_steps = 1
        self.seed = args.seed
        self.num_labels = args.num_labels
        self.label_list = [0, 1]
        self.task = args.task
        self.num_train_optimization_steps = \
            (num_train_examples + args.batch_size - 1) // args.batch_size * 3
        if args.adv_train:
            self.warmup_proportion = 0
            self.num_train_optimization_steps = 100000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bert_model = args.dir
        if os.path.exists(os.path.join(self.bert_model, "checkpoint")):
            with open(os.path.join(self.bert_model, "checkpoint")) as file:
                self.bert_model = os.path.join(self.bert_model, "ckpt-%d" % (int(file.readline())))
                print("BERT checkpoint:", self.bert_model)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.do_lower_case)
        self.softmax = torch.nn.Softmax(dim=-1)

        cache_dir = "cache/bert"
        self.model = BertForSequenceClassification.from_pretrained(self.bert_model,
                cache_dir=cache_dir,
                num_labels=self.num_labels)
        self.model.to(self.device)

        self._build_trainer()

    def _build_trainer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = BertAdam(optimizer_grouped_parameters,
            lr=self.learning_rate,
            warmup=self.warmup_proportion,
            t_total=self.num_train_optimization_steps)

    def save(self, epoch):
        # Save a trained model, configuration and tokenizer
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_dir = os.path.join(self.bert_model, "ckpt-%d" % epoch)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(output_dir)       

        print("BERT saved: %s" % output_dir) 

    def step(self, batch, is_train=False, requires_grad=False, attack=False):
        features = convert_examples_to_features(
            batch, self.label_list, self.max_seq_length, self.tokenizer)
        
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)
        label_ids = label_ids.to(self.device)

        if is_train:
            self.model.train()
            embeddings_output, logits = self.model(input_ids, segment_ids, input_mask)
        else:
            self.model.eval()
            if requires_grad:
                grad = torch.enable_grad()
            else:
                grad = torch.no_grad()
            with grad:
                embeddings_output, logits = self.model(input_ids, segment_ids, input_mask)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
        preds = self.softmax(logits)
        prob = preds
        error = 1 - preds[:, 0]    
        preds = torch.argmax(preds, dim=1)
        result = self.acc_and_f1(preds.detach().cpu().numpy(), label_ids.cpu().numpy())

        ret = [loss, result["acc"], preds, error]
        
        if is_train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()        

        return ret

    def simple_accuracy(self, preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(self, preds, labels):
        acc = self.simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }
