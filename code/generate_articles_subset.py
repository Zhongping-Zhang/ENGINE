import os
import argparse
import logging
import numpy as np
import json
import torch
from torch.utils.data.dataloader import DataLoader
# from transformers import (
#     GPT2LMHeadModel,
#     GPT2TokenizerFast,
# )

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    )

from dataloader import (
    get_dataset,
    get_dataset_cap, 
    get_dataset_NE_cap,
    get_visualnews,
    get_visualnews_cap,
    get_visualnews_NE_cap,
)
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

# MODEL_CLASSES = {
#     "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
# }

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)







parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path",default='log_base_goodnews',type=str,required=True,help="Path to pre-trained model")
parser.add_argument("--validation_file", type=str, default="")
parser.add_argument("--NE_path", type=str, default="", help="An optional input path to name entity folder.")
parser.add_argument("--include_art", type=bool, default=True, help="whether to include name entity of main articles")

parser.add_argument("--length", type=int, default=1024) # length
parser.add_argument("--stop_token", type=str, default='<|endofarticle|>', help="Token at which text generation is stopped")
parser.add_argument("--offset", type=int, default=0, help="number of prompt text in article")
parser.add_argument("--dataloader",type=str, default=None, help="specify the dataloader")
parser.add_argument("--block_size", type=int, default=1024)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
)
parser.add_argument(
    "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
)
parser.add_argument("--k", type=int, default=0) # 0 means donot use top-k sampling here
parser.add_argument("--p", type=float, default=0.96) # top-p sampling: \in {.9, .92, .94, .96, .98, 1.0}, default to 0.96
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
parser.add_argument("--remove_NE_labels", type=bool, default=False, help="whether to remove name entity labels")
parser.add_argument("--save_interval", type=int, default=500, help='interval to save generated article files')
args = parser.parse_args()


args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()

set_seed(args)


tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, pad_token_id=tokenizer.eos_token_id)
# tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name_or_path)
# model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, pad_token_id=tokenizer.eos_token_id)
model.to(args.device)


args.length = min(args.length, model.config.max_position_embeddings)
logger.info(args)







# =============================================================================
# """ 1. from prompt_text """
# prompt_text = "<|begindomain|>www.nytimes.com<|endofdomain|><|begindate|>10-05-2014<|endofdate|>\
#     <|begintitle|>We Want Privacy, but Can&#8217;t Stop Sharing <|endoftitle|>\
#         <|beginsummary|>News analysis; overlooked aspect in debate over privacy and the Internet is \
#         the psychic toll of lack of privacy; privacy research in both online and offline environments \
#         has shown that just the perception of being watched results in feelings of low self-esteem, \
#         depression and anxiety.<|endofsummary|>"
# encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt") 
# #'tf': Return TensorFlow tf.constant objects; 'pt': Return PyTorch torch.Tensor objects; 'np': Return Numpy np.ndarray objects.
# =============================================================================

name2loader = { 'goodnews_base': get_dataset,
                'goodnews_cap': get_dataset_cap,
                'goodnews_NE_cap': get_dataset_NE_cap,
                'visualnews_base': get_visualnews,
                'visualnews_cap': get_visualnews_cap,
                'visualnews_NE_cap': get_visualnews_NE_cap,
                }




""" 2. from DIDAN_article_list """
test_dataset = name2loader[args.dataloader](args, tokenizer, evaluate=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=data_collator,
                num_workers=0,
                )


ROOT = os.path.dirname(args.validation_file)
model_ROOT = args.model_name_or_path.split("/")[-1]+"_offset%d"%args.offset
os.makedirs(os.path.join(ROOT, model_ROOT),exist_ok=True)
article_ids = []
with open(args.validation_file,'r') as f:
    for line in f:
        article_ids.append(json.loads(line)['id'])
        
        
# offset = 100
NE_label_list = ["<|PERSON|>", "<|CARDINAL|>", "<|GPE|>", "<|ORG|>", "<|DATE|>", "<|NORP|>", 
                 "<|WORK_OF_ART|>", "<|FAC|>", "<|LOC|>", "<|ORDINAL|>", "<|EVENT|>", "<|MONEY|>",
                 "<|TIME|>", "<|PRODUCT|>", "<|QUANTITY|>", "<|PERCENT|>", "<|LAW|>", "<|LANGUAGE|>"]

num_samples = 0
num_batch = 0
generated_sequences = []

for encoded_data in tqdm(test_loader, desc='article generation'):
    article_begin_id = tokenizer.encode("<|beginarticle|>")[0]
    try:
        prompt_end_id = int(torch.where(encoded_data['input_ids']==article_begin_id)[1]) # 50270: <|beginarticle|>; 50271: <|endofarticle|>
    except:
        print("too long captions, set end id to 1000")
        prompt_end_id = 1000
        
    prompt_end_id = min(1024, prompt_end_id+args.offset)
    encoded_prompt = encoded_data['input_ids'][0:1,:prompt_end_id]
    prompt_text = tokenizer.decode(list(encoded_prompt.numpy()[0])) 
    #print(prompt_text)
    encoded_prompt = encoded_prompt.to(args.device) # (1,text_length)
    input_ids = encoded_prompt if encoded_prompt.size()[-1]!=0 else None
    
    output_sequences = model.generate(
        input_ids=input_ids,  # num_samples, prompt_text_length
        max_length=args.length, 
        do_sample=True,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_k=args.k,
        top_p=args.p,
        eos_token_id = tokenizer.encode(args.stop_token)[0], # "<|endofarticle|>"
        num_return_sequences=args.num_return_sequences,
    ) # output_sequences: (num_sequence, article_length)
    
    
    
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        if args.remove_NE_labels:
            filter_index = []
            for NE_label in NE_label_list:
                NE_id = tokenizer.encode(NE_label)
                filter_index.append(np.where(generated_sequence.cpu().numpy()==NE_id[0])[0])
            filter_index = np.concatenate(filter_index)
            generated_sequence = np.delete(generated_sequence.cpu().numpy(),filter_index)


        generated_sequence = generated_sequence.tolist()
    
        

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        
        # Remove all text after the stop token
        # text = text[: text.find(args.stop_token) if args.stop_token else None]
    
        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )
    
        generated_sequences.append(total_sequence)
    # break
    num_samples+=1
    if num_samples%args.save_interval==0:
        
        save_name = args.validation_file.split("/")[-1].split(".")[0]+"_result_batch%d.jsonl"%num_batch
        compensate = num_samples-len(generated_sequences)
        
        with open(os.path.join(ROOT,model_ROOT,save_name), "w") as outfile:
            for i in range(len(generated_sequences)):
                sequence = generated_sequences[i]
                article_id = article_ids[i+compensate]
                
                result = {article_id:sequence}
                json.dump(result,outfile)
                outfile.write("\n")
        
        generated_sequences = []
        num_batch+=1

    if num_samples == 1005:
        break





