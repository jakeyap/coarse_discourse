#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:04:58 2020

@author: jakeyap
"""
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
import time

from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer, BertModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

NUM_TO_PROCESS = 100000
#tokenizer = transformers.
#transformers.BertTokenizer
'''

'''
def tokenize_example():    
    print('Example usage of tokenizer')
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentence1 = 'hi there you slow poke.'
    sentence2 = 'i was here since twenty hours ago.'
    tokenized_sentence1 = tokenizer.tokenize(text=sentence1)
    tokenized_sentence2 = tokenizer.tokenize(text=sentence2)
    encoded_sentence = tokenizer.encode(text=tokenized_sentence1,
                                        text_pair=tokenized_sentence2,
                                        max_length=30, 
                                        pad_to_max_length=True)
    print(tokenizer.decode(encoded_sentence))    

def tokenize_and_encode_pairs(valid_comment_pairs, start=0, count=1e6):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_pairs = []
    token_type_ids = []
    attention_mask = []
    pair_labels = []
    counter = 0
    end = start + count
    for head, tail in valid_comment_pairs:
        if counter >= end:
            break
        
        if counter >= start:
            tokenized_head = tokenizer.tokenize(head['body'])
            tokenized_tail = tokenizer.tokenize(tail['body'])
            encoded_dict = tokenizer.encode_plus(text=tokenized_head,
                                                    text_pair=tokenized_tail,
                                                    max_length=128,
                                                    pad_to_max_length=True)
            encoded_pairs.append(encoded_dict['input_ids'])
            token_type_ids.append(encoded_dict['token_type_ids'])
            attention_mask.append(encoded_dict['attention_mask'])
            pair_labels.append([head['majority_type'], tail['majority_type']])
        
        counter = counter + 1
    return {'encoded_pairs': encoded_pairs,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'pair_labels': pair_labels}


if __name__ =='__main__':
    import reddit_utilities as reddit
    time_start = time.time()
    pairs, errors = reddit.flatten_threads2pairs_all('coarse_discourse_dump_reddit.json');
    valid_comment_pairs = reddit.filter_valid_pairs(pairs)

    data = tokenize_and_encode_pairs(valid_comment_pairs, count=NUM_TO_PROCESS)
    
    time_end = time.time()
    time_taken = time_end - time_start
    print('Time elapsed: %6.2fs' % time_taken)