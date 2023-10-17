#!/usr/bin/env python
# coding=utf-8
# Original work Copyright 2020 The HuggingFace Inc. team. All rights reserved.
# Modified work Copyright 2021 The authors of ENGINE: https://arxiv.org/pdf/2112.05917.pdf
# 
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
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
#from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import torch
import torch.nn.functional as F
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

from torch.utils.data.dataloader import DataLoader
from dataloader import get_dataset_NE_cap


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=True,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
    NE_path: Optional[str] = field(
    default=None,
    metadata={"help": "An optional input path to name entity folder."},
    )
    
    include_art: bool = field(
        default=True, metadata={"help": "Whether to include name entity of the main articles"}
    )
    
    def __post_init__(self):
        # if self.dataset_name is None and self.train_file is None and self.validation_file is None:
        #     raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt", 'jsonl'], "`train_file` should be a csv, a json(l) or a txt file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt", 'jsonl'], "`validation_file` should be a csv, a json(l) or a txt file."


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

print(training_args)
#assert training_args.eval_accumulation_steps
#print("steps is: ",training_args.eval_accumulation_steps)


# =============== 1. Setup logging ===============
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)

# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
logger.info(f"Training/evaluation parameters {training_args}")

# Set seed before initializing model.
set_seed(training_args.seed)




# =============== 3. config ===============
config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision,
    "use_auth_token": True if model_args.use_auth_token else None,
}
if model_args.model_name_or_path:
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
else:
    assert False, "need to specify the model name or path for testing!"

# =============== 4. tokenizer ===============
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
print("tokenizer length: ",len(tokenizer))

# =============== 5. model & add special tokens ===============
model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)


# =============== 2. get data ===============
if not data_args.block_size:
    data_args.block_size = tokenizer.model_max_length
else:
    data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

# Get datasets
train_dataset = get_dataset_NE_cap(data_args, tokenizer=tokenizer) if training_args.do_train else None
eval_dataset = get_dataset_NE_cap(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    

# # Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
)


def ind_where(array: np.ndarray, target, return_first_match=True, default_value=-1):
    """
    :param array: Single dimension array
    :param target: target to search for
    :param return_first_match: If true, return the first index that matches, otherwise, return the last one
    :param default_value: Index to return if there was no match
    :return: index of the first match, or -1 if nothing
    """
    assert array.ndim == 1
    matching_inds = np.where(array == target)[0]
    if len(matching_inds) > 0:
        if return_first_match:
            return int(matching_inds[0])
        else:
            return int(matching_inds[-1])
    return default_value


NE_label_list = ["<|PERSON|>", "<|CARDINAL|>", "<|GPE|>", "<|ORG|>", "<|DATE|>", "<|NORP|>", 
                 "<|WORK_OF_ART|>", "<|FAC|>", "<|LOC|>", "<|ORDINAL|>", "<|EVENT|>", "<|MONEY|>",
                 "<|TIME|>", "<|PRODUCT|>", "<|QUANTITY|>", "<|PERCENT|>", "<|LAW|>", "<|LANGUAGE|>"]



ppl_ex = []
ppl_ex_whole = []
special_count = 0
art_index_position = []

step = 64
for sub_idx in range(0,len(eval_dataset),step):
#    print(sub_idx)

    eval_dataset_subset = torch.utils.data.Subset(eval_dataset,np.arange(sub_idx,min(sub_idx+step,len(eval_dataset) )))
    outputs = trainer.predict(eval_dataset_subset)

    #predictions = np.argmax(outputs[0],axis=2)
    #gt_labels = outputs[1]

    for pred, gt in zip(outputs[0], outputs[1]):
        """filter out NE label tokens"""
        filter_index = []
        for NE_label in NE_label_list:
            NE_id = tokenizer.encode(NE_label)
            filter_index.append(np.where(gt==NE_id[0])[0])
        filter_index = np.concatenate(filter_index)
        
        gt = np.delete(gt,filter_index)# remove name entity labels 
        pred = np.delete(pred,filter_index-1, axis=0)
        
        
        # Omit the first token. Keep in mind input_ids is shifted by 1
        start_ind = ind_where(gt, target=tokenizer.encode("<|beginarticle|>")[0], default_value=0)
        end_ind = min(len(gt)-1, ind_where(gt, target=tokenizer.encode("<|endofarticle|>")[0], default_value=gt.shape[0]))
        
        if start_ind==0:
            special_count+=1
            continue # special case, no article from data loader
        
        art_index_position.append([start_ind,end_ind])
        
        
        
        pred_tensor = torch.from_numpy(pred[start_ind-1:end_ind]) # shift by 1
        gt_tensor = torch.from_numpy(gt[start_ind:end_ind+1])
        
        #compare_pred = np.argmax(pred[start_ind-1:end_ind],axis=1)
        #compare_gt = gt[start_ind:end_ind+1]
        
        loss = F.cross_entropy(pred_tensor,gt_tensor,reduction='mean').numpy()
        ppl_ex.append(np.float(loss))
        
        wholeloss = F.cross_entropy(torch.from_numpy(pred[:-1,:]), torch.from_numpy(gt[1:]), reduction='mean').numpy()
        ppl_ex_whole.append(np.float(wholeloss))
    
#ppl_ex = np.concatenate(ppl_ex)
ppl_ex = np.array(ppl_ex)
ppl_ex_whole = np.array(ppl_ex_whole)
print("Article perplexity is {:.3f}".format(np.exp(np.mean(ppl_ex))), flush=True)
print("Whole perplexity is {:.3f}".format(np.exp(np.mean(ppl_ex_whole))), flush=True)
print("# of special cases: ", special_count)


dir_name = os.path.dirname(data_args.validation_file)
#np.save(os.path.join(dir_name,'test_article_beginend_index.npy'),np.array(art_index_position))



logger.info("*** Evaluate ***")
metrics = trainer.evaluate()
max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
perplexity = math.exp(metrics["eval_loss"])
metrics["perplexity"] = perplexity

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

