import os
import time
import numpy as np
import pandas as pd
import tqdm
import torch

# import huggingface transformers
# requires transfromers 4.17.0
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os


class Blenderbot:
    
    def __init__(self, model_name = 'facebook/blenderbot-400M-distill', temperature=None, top_k=3, top_p=None, length_penalty=1.0, num_beams=1):
        """
        length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
                model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
                sequences.
                
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda")
        self.model = self.model.eval().to(self.device)
        
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.length_penalty = length_penalty
        self.num_beams = num_beams
    
    
    def get_response(self, message, reply_start=None, past=None, do_sample=True):
        with torch.no_grad():
            #tokenize the utterance
            inputs = self.tokenizer(message, return_tensors="pt").to(self.device)
            
            # TODO: take a reply
            if reply_start:
                reply_start = self.tokenizer(reply_start, return_tensors="pt")
                inputs['decoder_input_ids'] = reply_start['input_ids']
            
            #generate model results
            result = self.model.generate(**inputs, 
                                        temperature=self.temperature, 
                                        top_k=self.top_k,
                                        top_p=self.top_p,
                                        do_sample=do_sample,
                                        length_penalty=self.length_penalty,
                                        num_beams=self.num_beams
                                        )
        
        return self.tokenizer.decode(result[0]), None 
            