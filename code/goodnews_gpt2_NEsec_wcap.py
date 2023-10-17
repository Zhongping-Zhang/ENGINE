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

from arguments import (ModelArguments, DataTrainingArguments, TrainingArguments)
import transformers
# from transformers import (GPT2Tokenizer, GPT2TokenizerFast, GPT2Config, GPT2Model)
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from dataloader import get_dataset_NE_cap




########## 1. argument ##########
logger = logging.getLogger(__name__)

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# 1.1 Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

# Detecting last checkpoint.
last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    print("=========>>> LOAD last checkpoint ===========>>> ")
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

set_seed(training_args.seed)


##########################  2. tokenizer ####################################
tokenizer_kwargs = {
    "cache_dir": model_args.cache_dir, # specify cache_dir for pretrained model weights
    "use_fast": model_args.use_fast_tokenizer, # True
    "revision": model_args.model_revision, # 'main'
    "use_auth_token": True if model_args.use_auth_token else None, # False
}
if model_args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
elif model_args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )
    
print("initialized tokenizer length: ",len(tokenizer)) # 50257
additional_special_tokens=[]
for special_token_type in ['NE', 'date', 'title', 'captions', 'summary', 'article',]: # replace domain by NE
    additional_special_tokens.append('<|begin'+special_token_type+'|>')
    additional_special_tokens.append('<|endof'+special_token_type+'|>')
    
NE_label_list = ["<|PERSON|>", "<|CARDINAL|>", "<|GPE|>", "<|ORG|>", "<|DATE|>", "<|NORP|>", 
                 "<|WORK_OF_ART|>", "<|FAC|>", "<|LOC|>", "<|ORDINAL|>", "<|EVENT|>", "<|MONEY|>",
                 "<|TIME|>", "<|PRODUCT|>", "<|QUANTITY|>", "<|PERCENT|>", "<|LAW|>", "<|LANGUAGE|>"] # 18 categories
additional_special_tokens += NE_label_list    

special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'additional_special_tokens':additional_special_tokens}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print("specialized tokenizer length: ",len(tokenizer)) # 50272+18 = 50290

##########################  3. config ####################################
config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision, # main
    "use_auth_token": True if model_args.use_auth_token else None, # False
}
if model_args.config_name:
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
elif model_args.model_name_or_path:
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
else:
    config = CONFIG_MAPPING[model_args.model_type]()
    logger.warning("You are instantiating a new config instance from scratch.")


########################## 4. model #############################
model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

model.resize_token_embeddings(len(tokenizer))

######################## 4.2 model parallelize ##############
if model_args.model_name_or_path=='gpt2-xl':
    print("gpt2-xl")
    device_map = {0: [i for i in range(12)],
                  1: [i for i in range(12,24)],
                  2: [i for i in range(24,36)],
                  3: [i for i in range(36,48)]}
    model.parallelize(device_map)

elif model_args.model_name_or_path=='gpt2-medium':
    print("gpt2-medium")
    device_map = {0: [i for i in range(12)],
                  1: [i for i in range(12,24)]}
    model.parallelize(device_map)


######################## 5. data #########################
if not data_args.block_size: # None
    data_args.block_size = tokenizer.model_max_length # 1024
    # Our input block size will be the max possible for the model
else:
    data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

train_dataset = get_dataset_NE_cap(data_args, tokenizer=tokenizer) if training_args.do_train else None
eval_dataset = get_dataset_NE_cap(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


####################### 6. Trainer #########################
# # Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
)


# Training
if training_args.do_train:
    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    trainer.save_model()  # Saves the tokenizer too for easy upload
    tokenizer.save_pretrained(training_args.output_dir)
    
    
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# Evaluation
if training_args.do_eval:
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate()

    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    perplexity = math.exp(metrics["eval_loss"])
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

