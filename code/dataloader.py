import os
import ujson as json
from typing import Dict, List, Optional
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

class Goodnews_article_baseline(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int = 1024):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size
        self._init_dataset()
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    def _init_dataset(self):
        """save cuda memory, trade off a bit of speed for memory"""
        assert os.path.isfile(self.file_path), f"Input file path {file_path} not found"
        with open(self.file_path, 'r') as f:
            self.articles = []
            for l_no, l in enumerate(f):
                self.articles.append(json.loads(l))
        
        lines = []
        print("total number of news: ", len(self.articles))
        for article in tqdm(self.articles):
            article_pieces = {
            'article': article['text'], # 'article': "<|beginarticle|>"+article['text']+"<|endofarticle|>",
            }
            
            article_string = ''
            for piece_string in list(article_pieces.values()):
                article_string += piece_string
            
            lines.append(article_string)
        
        self.examples = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
        
        
        


class Goodnews_base(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int = 1024):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size
        self._init_dataset()
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    def _init_dataset(self):
        """save cuda memory, trade off a bit of speed for memory"""
        assert os.path.isfile(self.file_path), f"Input file path {file_path} not found"
        with open(self.file_path, 'r') as f:
            self.articles = []
            for l_no, l in enumerate(f):
                self.articles.append(json.loads(l))
        
        lines = []
        print("total number of news: ", len(self.articles))
        for article in tqdm(self.articles):
            article_pieces = {
            'domain': "<|begindomain|>"+article['domain']+"<|endofdomain|>",
            'date': "<|begindate|>"+article['publish_date']+"<|endofdate|>",
            'title': "<|begintitle|>"+article['title']+"<|endoftitle|>",
            'summary': "<|beginsummary|>"+article['summary']+"<|endofsummary|>" if article['summary'] else "<|beginsummary|>"+str(article['summary'])+"<|endofsummary|>",
            'article': "<|beginarticle|>"+article['text']+"<|endofarticle|>",
            }
            
            article_string = ''
            for piece_string in list(article_pieces.values()):
                article_string += piece_string
            
            lines.append(article_string)
        
        self.examples = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
    
    
class Goodnews_cap(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int = 1024):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size
        self._init_dataset()
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    def _init_dataset(self):
        assert os.path.isfile(self.file_path), f"Input file path {file_path} not found"

        with open(self.file_path, 'r') as f:
            self.articles = []
            for l_no, l in enumerate(f):
                self.articles.append(json.loads(l))
        
        lines = []
        print("total number of news: ", len(self.articles))
        for article in tqdm(self.articles):
            article_pieces = {
            'domain': "<|begindomain|>"+article['domain']+"<|endofdomain|>",
            'date': "<|begindate|>"+article['publish_date']+"<|endofdate|>",
            'title': "<|begintitle|>"+article['title']+"<|endoftitle|>",
            'captions': "<|begincaptions|>"+article['captioning']+"<|endofcaptions|>",
            'summary': "<|beginsummary|>"+article['summary']+"<|endofsummary|>" if article['summary'] else "<|beginsummary|>"+str(article['summary'])+"<|endofsummary|>",
            'article': "<|beginarticle|>"+article['text']+"<|endofarticle|>",
            }
            
            article_string = ''
            for piece_string in list(article_pieces.values()):
                article_string += piece_string
            
            lines.append(article_string)
            
        self.examples = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
    



"""gpt2-NE
remove domain address(little information)
add name entity information
"""
class Goodnews_NE_cap(Dataset):
    def __init__(self, tokenizer, file_path: str, NE_path: str, block_size: int = 1024, include_art = True):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.NE_path = NE_path
        self.block_size = block_size
        self.include_art = include_art
        if "train" in self.file_path:
            self.phase = "train"
        elif "test" in self.file_path: 
            self.phase = 'test'
        else:
            self.phase = 'val'

        self._init_dataset()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    def _init_dataset(self):
        assert os.path.isfile(self.file_path), f"Input file path {file_path} not found"

        with open(self.file_path, 'r') as f:
            self.articles = []
            for l_no, l in enumerate(f):
                self.articles.append(json.loads(l))
        
        with open(os.path.join(self.NE_path, "captionNE_%s.json"%self.phase),'r') as fc:
            caption2ne = json.load(fc)
        with open(os.path.join(self.NE_path, "titleNE_%s.json"%self.phase),'r') as ft:
            title2ne = json.load(ft)
        if self.include_art:
            with open(os.path.join(self.NE_path, "articleNE_%s.json"%self.phase), 'r') as fa:
                article2ne = json.load(fa)

        lines = []
        print("total number of news: ", len(self.articles))
        for article in tqdm(self.articles):
            art_id = article['id']
            NE_cap = caption2ne[art_id]
            NE_title = title2ne[art_id]
            NE_capstr = ''
            NE_titlestr = ''

            for key,value in NE_cap.items():
                NE_capstr += "<|%s|>"%key+' '.join(value)
            for key,value in NE_title.items():
                NE_titlestr += "<|%s|>"%key+' '.join(value)

            NE_whole = NE_titlestr+"\n"+NE_capstr
            if self.include_art:
                NE_art = article2ne[art_id]
                NE_artstr = ''
                for key,value in NE_art.items():
                    NE_artstr += "<|%s|>"%key+' '.join(value)
                NE_whole+="\n"+NE_artstr
            #print(NE_whole)

            article_pieces = {
            #'domain': "<|begindomain|>"+article['domain']+"<|endofdomain|>",
            'date': "<|begindate|>"+article['publish_date']+"<|endofdate|>",
            'nameentity': "<|beginNE|>"+NE_whole+"<|endofNE|>",
            'title': "<|begintitle|>"+article['title']+"<|endoftitle|>",
            'captions': "<|begincaptions|>"+article['captioning']+"<|endofcaptions|>",
            'summary': "<|beginsummary|>"+article['summary']+"<|endofsummary|>" if article['summary'] else "<|beginsummary|>"+str(article['summary'])+"<|endofsummary|>",
            'article': "<|beginarticle|>"+article['text']+"<|endofarticle|>",
            }
            
            article_string = ''
            for piece_string in list(article_pieces.values()):
                article_string += piece_string
            
            lines.append(article_string)
            
        self.examples = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]


class Goodnews_ClipNE_cap(Dataset):
    def __init__(self, tokenizer, file_path: str, NE_path: str, block_size: int = 1024, include_art = True):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.NE_path = NE_path
        self.block_size = block_size
        self.include_art = include_art
        if "train" in self.file_path:
            self.phase = "train"
        elif "test" in self.file_path: 
            self.phase = 'test'
        else:
            self.phase = 'val'

        self._init_dataset()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    def _init_dataset(self):
        assert os.path.isfile(self.file_path), f"Input file path {file_path} not found"

        with open(self.file_path, 'r') as f:
            self.articles = []
            for l_no, l in enumerate(f):
                self.articles.append(json.loads(l))
        
        with open(os.path.join(self.NE_path, "imageNE_%s.json"%self.phase),'r') as fi:
            image2ne = json.load(fi)


        lines = []
        print("total number of news: ", len(self.articles))
        for article in tqdm(self.articles):
            art_id = article['id']
            NE_image = image2ne[art_id]['name_entity']
            NE_imgstr = ' '.join(NE_image)

            NE_whole = NE_imgstr
            # print(NE_whole)

            article_pieces = {
            'date': "<|begindate|>"+article['publish_date']+"<|endofdate|>",
            'nameentity': "<|beginNE|>"+NE_whole+"<|endofNE|>",
            'title': "<|begintitle|>"+article['title']+"<|endoftitle|>",
            'captions': "<|begincaptions|>"+article['captioning']+"<|endofcaptions|>",
            'summary': "<|beginsummary|>"+article['summary']+"<|endofsummary|>" if article['summary'] else "<|beginsummary|>"+str(article['summary'])+"<|endofsummary|>",
            'article': "<|beginarticle|>"+article['text']+"<|endofarticle|>",
            }
            
            article_string = ''
            for piece_string in list(article_pieces.values()):
                article_string += piece_string
            
            lines.append(article_string)
            
        self.examples = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
        
        

"""goodnews dataloader functions"""    
def get_dataset(args, tokenizer, evaluate=False):
    file_path = args.validation_file if evaluate else args.train_file
    return Goodnews_base(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
def get_dataset_cap(args, tokenizer, evaluate=False):
    file_path = args.validation_file if evaluate else args.train_file
    return Goodnews_cap(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
def get_dataset_NE_cap(args, tokenizer, evaluate=False):
    file_path = args.validation_file if evaluate else args.train_file
    return Goodnews_NE_cap(tokenizer=tokenizer, file_path=file_path, NE_path=args.NE_path, block_size=args.block_size, include_art=args.include_art)
def get_dataset_ClipNE_cap(args, tokenizer, evaluate=False):
    file_path = args.validation_file if evaluate else args.train_file
    return Goodnews_ClipNE_cap(tokenizer=tokenizer, file_path=file_path, NE_path=args.NE_path, block_size=args.block_size)

def get_goodnews_baseline(args, tokenizer, evaluate=False):
    file_path = args.validation_file if evaluate else args.train_file
    return Goodnews_article_baseline(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)




import time
class Visualnews_article_baseline(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int = 1024):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size
        self._init_dataset()
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    def _init_dataset(self):
        """save cuda memory, trade off a bit of speed for memory"""
        assert os.path.isfile(self.file_path), f"Input file path {file_path} not found"
        with open(self.file_path, 'r') as f:
            self.articles = []
            for l_no, l in enumerate(f):
                self.articles.append(json.loads(l))
        
        lines = []
        num_notitle = 0
        notitle_dict = []
        print("total number of news: ", len(self.articles))
        for article in tqdm(self.articles):
            if not article['title']:
                num_notitle+=1
                notitle_dict.append(article['domain'])
                continue
            
            article_pieces = {
            'article': article['text'],
            }
            
            article_string = ''
            for piece_string in list(article_pieces.values()):
                article_string += piece_string
            
            lines.append(article_string)
        
        start_time = time.time()
        print("there are %d articles without title"%num_notitle)
        self.examples = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
        
        end_time = time.time()
        print("time to process articles: %d second"%(end_time-start_time))
        

class Visualnews_base(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int = 1024):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size
        self._init_dataset()
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    def _init_dataset(self):
        """save cuda memory, trade off a bit of speed for memory"""
        assert os.path.isfile(self.file_path), f"Input file path {file_path} not found"
        with open(self.file_path, 'r') as f:
            self.articles = []
            for l_no, l in enumerate(f):
                self.articles.append(json.loads(l))
        
        lines = []
        num_notitle = 0
        notitle_dict = []
        print("total number of news: ", len(self.articles))
        for article in tqdm(self.articles):
            if not article['title']:
                num_notitle+=1
                notitle_dict.append(article['domain'])
                continue
            
            article_pieces = {
            'domain': "<|begindomain|>"+article['domain']+"<|endofdomain|>", # not much helpful
            'date': "<|begindate|>"+article['publish_date']+"<|endofdate|>", # not much helpful
            'topic': "<|begintopic|>"+article['topic']+"<|endoftopic|>",
            'title': "<|begintitle|>"+article['title']+"<|endoftitle|>",
            'article': "<|beginarticle|>"+article['text']+"<|endofarticle|>",
            }
            
            article_string = ''
            for piece_string in list(article_pieces.values()):
                article_string += piece_string
            
            lines.append(article_string)
        
        start_time = time.time()
        print("there are %d articles without title"%num_notitle)
        self.examples = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
        
        end_time = time.time()
        print("time to process articles: %d second"%(end_time-start_time))


class Visualnews_cap(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int = 1024):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size
        self._init_dataset()
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    def _init_dataset(self):
        """save cuda memory, trade off a bit of speed for memory"""
        assert os.path.isfile(self.file_path), f"Input file path {file_path} not found"
        with open(self.file_path, 'r') as f:
            self.articles = []
            for l_no, l in enumerate(f):
                self.articles.append(json.loads(l))
        
        lines = []
        print("total number of news: ", len(self.articles))
        for article in tqdm(self.articles):
            
            article_pieces = {
            'domain': "<|begindomain|>"+article['domain']+"<|endofdomain|>", # not much helpful
            'date': "<|begindate|>"+article['publish_date']+"<|endofdate|>", # not much helpful
            'topic': "<|begintopic|>"+article['topic']+"<|endoftopic|>",
            'title': "<|begintitle|>"+article['title']+"<|endoftitle|>",
            'captions': "<|begincaptions|>"+article['captioning']+"<|endofcaptions|>",
            'article': "<|beginarticle|>"+article['text']+"<|endofarticle|>",
            }
            
            article_string = ''
            for piece_string in list(article_pieces.values()):
                article_string += piece_string
            
            lines.append(article_string)
        
        start_time = time.time()
        self.examples = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
        
        end_time = time.time()
        print("time to process articles: %d second"%(end_time-start_time))

class Visualnews_NE_cap(Dataset):
    def __init__(self, tokenizer, file_path: str, NE_path: str, block_size: int = 1024, include_art = True):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.NE_path = NE_path
        self.block_size = block_size
        self.include_art = include_art
        if "train" in self.file_path:
            self.phase = "train"
        elif "test" in self.file_path: 
            self.phase = 'test'
        else:
            self.phase = 'val'

        self._init_dataset()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    def _init_dataset(self):
        assert os.path.isfile(self.file_path), f"Input file path {self.file_path} not found"

        with open(self.file_path, 'r') as f:
            self.articles = []
            for l_no, l in enumerate(f):
                self.articles.append(json.loads(l))
        
        with open(os.path.join(self.NE_path, "captionNE_%s.json"%self.phase),'r') as fc:
            caption2ne = json.load(fc)
        with open(os.path.join(self.NE_path, "titleNE_%s.json"%self.phase),'r') as ft:
            title2ne = json.load(ft)
        if self.include_art:
            with open(os.path.join(self.NE_path, "articleNE_%s.json"%self.phase), 'r') as fa:
                article2ne = json.load(fa)

        lines = []
        print("total number of news: ", len(self.articles))
        for article in tqdm(self.articles):
            art_id = article['unique_id']
            NE_cap = caption2ne[art_id]
            NE_title = title2ne[art_id]
            NE_capstr = ''
            NE_titlestr = ''

            for key,value in NE_cap.items():
                NE_capstr += "<|%s|>"%key+' '.join(value)
            for key,value in NE_title.items():
                NE_titlestr += "<|%s|>"%key+' '.join(value)

            NE_whole = NE_titlestr+"\n"+NE_capstr
            if self.include_art:
                NE_art = article2ne[art_id]
                NE_artstr = ''
                for key,value in NE_art.items():
                    NE_artstr += "<|%s|>"%key+' '.join(value)
                NE_whole+="\n"+NE_artstr
            #print(NE_whole)

            article_pieces = {
            'domain': "<|begindomain|>"+article['domain']+"<|endofdomain|>",
            'date': "<|begindate|>"+article['publish_date']+"<|endofdate|>",
            'topic': "<|begintopic|>"+article['topic']+"<|endoftopic|>",
            'nameentity': "<|beginNE|>"+NE_whole+"<|endofNE|>",
            'title': "<|begintitle|>"+article['title']+"<|endoftitle|>",
            'captions': "<|begincaptions|>"+article['captioning']+"<|endofcaptions|>",
            'article': "<|beginarticle|>"+article['text']+"<|endofarticle|>",
            }
            
            article_string = ''
            for piece_string in list(article_pieces.values()):
                article_string += piece_string
            
            lines.append(article_string)
            
        self.examples = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]




class Visualnews_ClipNE_cap(Dataset):
    def __init__(self, tokenizer, file_path: str, NE_path: str, block_size: int = 1024, include_art = True):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.NE_path = NE_path
        self.block_size = block_size
        self.include_art = include_art
        if "train" in self.file_path:
            self.phase = "train"
        elif "test" in self.file_path: 
            self.phase = 'test'
        else:
            self.phase = 'val'

        self._init_dataset()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    def _init_dataset(self):
        assert os.path.isfile(self.file_path), f"Input file path {file_path} not found"

        with open(self.file_path, 'r') as f:
            self.articles = []
            for l_no, l in enumerate(f):
                self.articles.append(json.loads(l))
        
        with open(os.path.join(self.NE_path, "imageNE_%s.json"%self.phase),'r') as fi:
            image2ne = json.load(fi)


        lines = []
        print("total number of news: ", len(self.articles))
        for article in tqdm(self.articles):
            art_id = article['unique_id']
            NE_image = image2ne[art_id]['name_entity']
            NE_imgstr = ' '.join(NE_image)

            NE_whole = NE_imgstr
            # print(NE_whole)

            article_pieces = {
            'domain': "<|begindomain|>"+article['domain']+"<|endofdomain|>",
            'date': "<|begindate|>"+article['publish_date']+"<|endofdate|>",
            'topic': "<|begintopic|>"+article['topic']+"<|endoftopic|>",
            'nameentity': "<|beginNE|>"+NE_whole+"<|endofNE|>",
            'title': "<|begintitle|>"+article['title']+"<|endoftitle|>",
            'captions': "<|begincaptions|>"+article['captioning']+"<|endofcaptions|>",
            'article': "<|beginarticle|>"+article['text']+"<|endofarticle|>",
            }
            
            article_string = ''
            for piece_string in list(article_pieces.values()):
                article_string += piece_string
            
            lines.append(article_string)
            
        self.examples = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
        
        
class Visualnews_NE_cap_source(Dataset):
    def __init__(self, tokenizer, file_path: str, NE_path: str, block_size: int = 1024, include_art = True, source='bbc'):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.NE_path = NE_path
        self.block_size = block_size
        self.include_art = include_art
        self.source=source
        if "train" in self.file_path:
            self.phase = "train"
        elif "test" in self.file_path: 
            self.phase = 'test'
        else:
            self.phase = 'val'

        self._init_dataset()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    def _init_dataset(self):
        assert os.path.isfile(self.file_path), f"Input file path {file_path} not found"

        with open(self.file_path, 'r') as f:
            self.articles = []
            for l_no, l in enumerate(f):
                self.articles.append(json.loads(l))
        
        with open(os.path.join(self.NE_path, "captionNE_%s.json"%self.phase),'r') as fc:
            caption2ne = json.load(fc)
        with open(os.path.join(self.NE_path, "titleNE_%s.json"%self.phase),'r') as ft:
            title2ne = json.load(ft)
        if self.include_art:
            with open(os.path.join(self.NE_path, "articleNE_%s.json"%self.phase), 'r') as fa:
                article2ne = json.load(fa)

        lines = []
        print("total number of news: ", len(self.articles))
        for article in tqdm(self.articles):
            art_id = article['unique_id']
            NE_cap = caption2ne[art_id]
            NE_title = title2ne[art_id]
            NE_capstr = ''
            NE_titlestr = ''

            if self.source not in article['domain']:
                continue

            for key,value in NE_cap.items():
                NE_capstr += "<|%s|>"%key+' '.join(value)
            for key,value in NE_title.items():
                NE_titlestr += "<|%s|>"%key+' '.join(value)

            NE_whole = NE_titlestr+"\n"+NE_capstr
            if self.include_art:
                NE_art = article2ne[art_id]
                NE_artstr = ''
                for key,value in NE_art.items():
                    NE_artstr += "<|%s|>"%key+' '.join(value)
                NE_whole+="\n"+NE_artstr
            #print(NE_whole)

            article_pieces = {
            'domain': "<|begindomain|>"+article['domain']+"<|endofdomain|>",
            'date': "<|begindate|>"+article['publish_date']+"<|endofdate|>",
            'topic': "<|begintopic|>"+article['topic']+"<|endoftopic|>",
            'nameentity': "<|beginNE|>"+NE_whole+"<|endofNE|>",
            'title': "<|begintitle|>"+article['title']+"<|endoftitle|>",
            'captions': "<|begincaptions|>"+article['captioning']+"<|endofcaptions|>",
            'article': "<|beginarticle|>"+article['text']+"<|endofarticle|>",
            }
            
            article_string = ''
            for piece_string in list(article_pieces.values()):
                article_string += piece_string
            
            lines.append(article_string)
        print("evaluation dataset length: ", len(lines))
        self.examples = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]




def get_visualnews(args, tokenizer, evaluate=False):
    file_path = args.validation_file if evaluate else args.train_file
    return Visualnews_base(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
def get_visualnews_cap(args, tokenizer, evaluate=False):
    file_path = args.validation_file if evaluate else args.train_file
    return Visualnews_cap(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
def get_visualnews_NE_cap(args, tokenizer, evaluate=False):
    file_path = args.validation_file if evaluate else args.train_file
    return Visualnews_NE_cap(tokenizer=tokenizer, file_path=file_path, NE_path=args.NE_path, block_size=args.block_size, include_art=args.include_art)
def get_visualnews_ClipNE_cap(args, tokenizer, evaluate=False):
    file_path = args.validation_file if evaluate else args.train_file
    return Visualnews_ClipNE_cap(tokenizer=tokenizer, file_path=file_path, NE_path=args.NE_path, block_size=args.block_size)

def get_visualnews_baseline(args, tokenizer, evaluate=False):
    file_path = args.validation_file if evaluate else args.train_file
    return Visualnews_article_baseline(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)

def get_visualnews_NE_cap_source(args, tokenizer, evaluate=False, source='bbc'):
    file_path = args.validation_file if evaluate else args.train_file
    return Visualnews_NE_cap_source(tokenizer=tokenizer, file_path=file_path, NE_path=args.NE_path, block_size=args.block_size, include_art=args.include_art, source=source)
# source list: ['guardian', 'washington_post', 'bbc', 'usa_today']



if __name__=="__main__":
    pass