#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:04:58 2020

@author: jakeyap
"""
import logging
import time

import reddit_utilities as reddit
from transformers import BertTokenizer

import torch
from torch.utils.data import TensorDataset, DataLoader
'''
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
'''

#tokenizer = transformers.
#transformers.BertTokenizer

def convert_label_pairs_string2num(label1, label2):
    ''' 
    Converts pairwise text labels into numbers
    label1 x 10 + label2
    '''
    dictionary = reddit.empty_label_dictionary()
    all_labels = list(dictionary.keys())
    num1 = all_labels.index(label1)
    num2 = all_labels.index(label2)
    
    pair_label = num1*10 + num2
    return pair_label
    
def convert_label_pairs_num2string(pair_label):
    '''
    Converts pairwise number labels back into text labels
    The forward operation is : label1 x 10 + label2
    '''
    dictionary = reddit.empty_label_dictionary()
    labels = list(dictionary.keys())
    
    num1 = pair_label // 10 # divide and take quotient
    num2 = pair_label % 10  # divide and take remainder
    label1 = labels[num1]
    label2 = labels[num2]
    return [label1, label2]

def tokenize_example():    
    print('Example usage of tokenizer')
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
        if counter % 1000 == 0:
                print('Tokenizing comment: %00000d' % counter)
        counter = counter + 1
    return {'encoded_pairs': encoded_pairs,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'pair_labels': pair_labels}

def split_dict_2_train_test_sets(data_dict, test_percent, 
                                 training_batch_size=64,
                                 testing_batch_size=64,
                                 randomize=False,
                                 device='cuda'):
    '''
    Takes in a data dictionary, then splits them into 2 dictionaries
    
    data_dict is a dictionary which contains all examples. 
    {
        'encoded_pairs': encoded_pairs,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'pair_labels': pair_labels
    }
    
    Returns a list of 2 Dataloaders. [train_loader, tests_loader]
    Each dataloader is packed into (x, y, token_type_ids, attention_mask)
    '''
    pair_labels = data_dict['pair_labels']
    pair_labels_num = []
    for each_pair in pair_labels:
        number_label = convert_label_pairs_string2num(each_pair[0],each_pair[1])
        pair_labels_num.append(number_label)
    
    x_data = torch.tensor(data_dict['encoded_pairs'])
    y_data = torch.tensor(pair_labels_num)
    token_type_ids = torch.tensor(data_dict['token_type_ids'])
    attention_mask = torch.tensor(data_dict['attention_mask'])
    
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    if randomize:
        pass # randomly shuffle test set selection. not implemented yet
    
    datalength = y_data.shape[0]
    stopindex = int (datalength * (100 - test_percent) / 100)
    x_train = x_data [0:stopindex]
    y_train = y_data [0:stopindex]
    token_type_ids_train = token_type_ids [0:stopindex]
    attention_mask_train = attention_mask [0:stopindex]
    
    x_tests = x_data [stopindex:]
    y_tests = y_data [stopindex:]
    token_type_ids_tests = token_type_ids [stopindex:]
    attention_mask_tests = attention_mask [stopindex:]
    
    train_dataset = TensorDataset(x_train,
                                  y_train,
                                  token_type_ids_train,
                                  attention_mask_train)
    tests_dataset = TensorDataset(x_tests,
                                  y_tests,
                                  token_type_ids_tests,
                                  attention_mask_tests)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=training_batch_size,
                              shuffle=True)
    tests_loader = DataLoader(tests_dataset,
                              batch_size=testing_batch_size)
    
    return [train_loader, tests_loader]

if __name__ =='__main__':
    
    import reddit_utilities as reddit
    time_start = time.time()
    NUM_TO_PROCESS = 1000
    pairs, errors = reddit.flatten_threads2pairs_all('coarse_discourse_dump_reddit.json');
    valid_comment_pairs = reddit.filter_valid_pairs(pairs)

    data = tokenize_and_encode_pairs(valid_comment_pairs, count=NUM_TO_PROCESS)
    
    time_end = time.time()
    time_taken = time_end - time_start
    print('Time elapsed: %6.2fs' % time_taken)
    