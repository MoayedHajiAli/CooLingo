import os
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# import huggingface transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW
import os



class DialoGPT:
    
    models_uris = {
        'small':'https://acvrpublicycchen.blob.core.windows.net/dialogpt/multiref/small_ft.pkl',
        'medium':'https://acvrpublicycchen.blob.core.windows.net/dialogpt/multiref/medium_ft.pkl',
        'large':'https://acvrpublicycchen.blob.core.windows.net/dialogpt/multiref/large_ft.pkl'
    }
    models_config = {
        'small' : {},
        'medium' : {'n_ctx':1024, 'n_embd':1024, 'n_layer':24, 'n_head':16},
        'large' : {'n_ctx':1024, 'n_embd':1280, 'n_layer':36, 'n_head':20},
    }
    
    def __init__(self, model_size='large', temperature=0.5, top_k=5, top_p=0.8):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = self.get_model(model_size)
        self.device = torch.device("cuda")
        self.model = self.model.to(self.device)
        self.model.lm_head.weight.data = self.model.transformer.wte.weight.data
        
        self.eos = [self.tokenizer.encoder["<|endoftext|>"]]
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
    
    def get_model(self, model_size):
        assert model_size in self.models_uris.keys(), "Given model_size {model_size} is unidentified"
        
        model_local_path = f"pretrained_models/dialogpt_{model_size}.pkl"
        gpt2_config = GPT2Config(**self.models_config[model_size])
        
        if not os.path.exists(model_local_path):
            print(f"Donwloading pretrained model to {model_local_path}")
            os.system(f"wget {self.models_uris[model_size]} -O {model_local_path}")
        
        model = GPT2LMHeadModel(gpt2_config)
        model.load_state_dict(torch.load(model_local_path), strict=False)
        
        return model
        
        
    def top_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ 
        Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p
        """
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
            logits = torch.where(logits < min_values, 
                                 torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                                 logits)
        if top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
            logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

        return logits

    
    def get_response(self, message, past=None, ):
        with torch.no_grad():
            message = self.tokenizer.encode(message)
            prev_input = message
            prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device)
            _, past = self.model(prev_input, past=past)

            prev_input = torch.LongTensor([self.eos]).to(self.device)

            sent = []
            for i in range(500):
                logits, past = self.model(prev_input, past=past)
                logits = logits[:, -1, :] / self.temperature
                logits = self.top_filtering(logits, top_k=self.top_k, top_p=self.top_p)

                probs = torch.softmax(logits, dim=-1)

                prev_input = torch.multinomial(probs, num_samples=1)
                prev_word = prev_input.item()

                if prev_word == self.eos[0]:
                    break
                sent.append(prev_word)

            response = self.tokenizer.decode(sent)
            prev_input = torch.LongTensor([self.eos]).to(self.device)
            _, past = self.model(prev_input, past=past)
        
        return response, past