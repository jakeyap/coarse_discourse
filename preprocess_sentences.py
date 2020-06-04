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
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
'''
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
'''
dictionary = reddit.empty_label_dictionary()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def convert_label_pairs_string2num(label1, label2):
    ''' 
    Converts pairwise text labels into numbers
    label1 x 10 + label2
    '''
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
    labels = list(dictionary.keys())
    
    num1 = pair_label // 10 # divide and take quotient
    num2 = pair_label % 10  # divide and take remainder
    label1 = labels[num1]
    label2 = labels[num2]
    return [label1, label2]

def convert_label_string2num(label):
    '''
    Converts text label into a number
    '''
    all_labels = list(dictionary.keys())
    if label == 'firstpost':
        return len(all_labels)
    else:
        return all_labels.index(label)
    
def convert_label_num2string(number):
    '''
    Converts a numerical label back into a string
    '''
    all_labels = list(dictionary.keys())
    if number == len(all_labels):
        return 'firstpost'
    else:
        return all_labels[number]

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

def tokenize_and_encode_pairs(valid_comment_pairs, start=0, count=1e6, tk=None):
    
    if tk is None:
        tk = BertTokenizer.from_pretrained('bert-base-uncased')
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
            tokenized_head = tk.tokenize(head['body'])
            tokenized_tail = tk.tokenize(tail['body'])
            encoded_dict = tk.encode_plus(text=tokenized_head,
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
                                 device='cpu'):
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
        print('Shuffling data')
        shuffle_arrays_synch([x_data, y_data, token_type_ids, attention_mask])
    
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
                              shuffle=randomize)
    tests_loader = DataLoader(tests_dataset,
                              batch_size=testing_batch_size)
    
    return [train_loader, tests_loader]

def shuffle_arrays_synch(arrays, set_seed=-1):
    """Shuffles arrays in-place, in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of torch tensors.
    set_seed : Seed value if int >= 0, else seed is random.
    """
    # set up the seed for the RNG
    seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed
    torch.manual_seed(seed)
    # number of torch tensors inside arrays
    try:
        datalength = arrays[0].shape[0]
    except Exception:
        datalength = len(arrays[0])
    # set up the shuffling indexes
    shuffle_indices = torch.randint(low=0, high=datalength, size=(datalength,1))
    
    for arr in arrays:
        for i in range(datalength):
            j = shuffle_indices[i]
            temp = torch.clone(arr[i])
            arr[i] = torch.clone(arr[j])
            arr[j] = temp

def tokenize_and_encode_comments(valid_comments, start=0, count=1e6, tk=None):
    if tk is None:
        tk = BertTokenizer.from_pretrained('bert-base-uncased')
    
    encoded_comments = []
    token_type_ids = []
    attention_mask = []
    labels = []
    parent_ids = []
    counter = 0
    end = start + count
    for each_comment in valid_comments:
        if counter >= end:
            break
        
        if counter >= start:
            tokenized_comment = tk.tokenize(each_comment['body'])
            #print(tokenized_comment)
            encoded_dict = tk.encode_plus(text=tokenized_comment,
                                          max_length=128,
                                          pad_to_max_length=True)
            #print(encoded_dict)
            encoded_comments.append(encoded_dict['input_ids'])
            token_type_ids.append(encoded_dict['token_type_ids'])
            attention_mask.append(encoded_dict['attention_mask'])
            labels.append(each_comment['majority_type'])
            if 'majority_link' in each_comment.keys():
                parent_ids.append(each_comment['majority_link'])
            elif 'in_reply_to' in each_comment.keys():
                parent_ids.append(each_comment['in_reply_to'])
            else:
                parent_ids.append('nil')
        if counter % 1000 == 0:
            print('Tokenizing comment: %00000d' % counter)
        counter = counter + 1
    return {'encoded_comments': encoded_comments,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'parent_ids':parent_ids}

def tokenize_and_encode_pandas(dataframe, tk=None):
    if tk is None:
        tk = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_comments = []
    token_type_ids = []
    attention_mask = []
    labels = []
    parent_labels = []
    counter = 0
    for i in range(len(dataframe)):
        tokenized_comment = tk.tokenize(dataframe.at[i, 'body'])
        encoded_dict = tk.encode_plus(text=tokenized_comment,
                                      max_length=128,
                                      pad_to_max_length=True)
        encoded_comments.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_mask.append(encoded_dict['attention_mask'])
        
        label = dataframe.at[i, 'majority_type']
        labels.append(convert_label_string2num(label))
        parent_id   = dataframe.at[i, 'majority_link']
        parent_index= reddit.pandas_find_parent_index(parent_id, dataframe)
        if parent_index==-1:
            parent_label = 'firstpost'
        else:
            parent_label = dataframe.at[parent_index, 'majority_type']
        parent_labels.append(convert_label_string2num(parent_label))
        
        if counter % 1000 == 0:
            print('Tokenizing comment: %00000d' % counter)
        counter = counter + 1
    
    width = dataframe.shape[1]
    dataframe.insert(width+0, 'encoded_comments', encoded_comments)
    dataframe.insert(width+1,'token_type_ids', token_type_ids)
    dataframe.insert(width+2,'attention_mask', attention_mask)
    dataframe.insert(width+3,'number_labels', labels)
    dataframe.insert(width+4,'parent_labels', parent_labels)
    return dataframe
    
def split_dict_2_train_test_sets_single(data_dict, test_percent, 
                                        training_batch_size=64,
                                        testing_batch_size=64,
                                        randomize=False,
                                        device='cpu'):
    '''
    Takes in a data dictionary, then splits them into 2 dictionaries
    
    data_dict is a dictionary which contains all examples. 
    {
        'encoded_comments': encoded_comments,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
    
    Returns a list of 2 Dataloaders. [train_loader, tests_loader]
    Each dataloader is packed into (x, y, token_type_ids, attention_mask)
    '''
    labels = data_dict['labels']
    labels_num = []
    for each_label in labels:
        number_label = convert_label_string2num(each_label)
        labels_num.append(number_label)
    
    x_data = torch.tensor(data_dict['encoded_comments'])
    y_data = torch.tensor(labels_num)
    token_type_ids = torch.tensor(data_dict['token_type_ids'])
    attention_mask = torch.tensor(data_dict['attention_mask'])
    
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    if randomize:
        print('Shuffling data')
        shuffle_arrays_synch([x_data, y_data, 
                              token_type_ids, 
                              attention_mask])
    
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
                              shuffle=randomize)
    tests_loader = DataLoader(tests_dataset,
                              batch_size=testing_batch_size)
    
    return [train_loader, tests_loader]

def split_pandas_2_train_test_sets(dataframe,
                                   test_percent, 
                                   training_batch_size=64,
                                   testing_batch_size=64,
                                   randomize=False,
                                   DEBUG=False):
    '''Splits up dataframe into 2. One for test, one for train'''
    """
    Takes in a pandas dataframe, then splits them into 2 dictionaries
    
    dataframe is a pandas dataframe which contains all examples. 
    it has the following columns 
    {
        is_first_post, id, encoded_comments, majority_type, url, author,
        majority_link, body, annotations, post_depth, in_reply_to,
        encoded_comments, token_type_ids, attention_masks, 
        number_labels, parent_labels
    }
    
    Returns a list of 2 Dataloaders. [train_loader, tests_loader]
    Each dataloader is packed into the following tuple
    {   
         index in original data,
         x (encoded comment), 
         token_typed_ids,
         attention_masks,
         y (true label),
         parent_y (parent's label)
    }
    """
    if DEBUG:
        pass
        datalength = 1200
        midindex = int (datalength * (100 - test_percent) / 100)
        stopindex = datalength
    else:
        datalength = len(dataframe)
        midindex = int (datalength * (100 - test_percent) / 100)
        stopindex = datalength
    
    posts_index     = np.arange(0, datalength)
    
    encoded_comments= dataframe['encoded_comments'].values
    encoded_comments= np.array(encoded_comments.tolist())
    token_type_ids  = dataframe['token_type_ids'].values
    token_type_ids  = np.array(token_type_ids.tolist())
    attention_masks = dataframe['attention_masks'].values
    attention_masks = np.array(attention_masks.tolist())
    
    number_labels   = dataframe['number_labels'].values
    parent_labels   = dataframe['parent_labels'].values
    
    posts_index     = posts_index.reshape(((-1,1)))
    
    number_labels   = number_labels.reshape(((-1,1)))
    parent_labels   = parent_labels.reshape(((-1,1)))
    
    # convert numpy arrays into torch tensors
    posts_index     = torch.from_numpy(posts_index)
    encoded_comments= torch.from_numpy(encoded_comments)
    token_type_ids  = torch.from_numpy(token_type_ids)
    attention_masks = torch.from_numpy(attention_masks)
    number_labels   = torch.from_numpy(number_labels)
    parent_labels   = torch.from_numpy(parent_labels)
    
    if randomize:
        # Do shuffle here
        shuffle_arrays_synch([posts_index,
                              encoded_comments,
                              token_type_ids,
                              attention_masks,
                              number_labels,
                              parent_labels])
        
    train_posts_index       = posts_index[0:midindex]
    train_encoded_comments  = encoded_comments[0:midindex]
    train_token_type_ids    = token_type_ids[0:midindex]
    train_attention_masks   = attention_masks[0:midindex]
    train_number_labels     = number_labels[0:midindex]
    train_parent_labels     = parent_labels[0:midindex]
    
    tests_posts_index       = posts_index[midindex:stopindex]
    tests_encoded_comments  = encoded_comments[midindex:stopindex]
    tests_token_type_ids    = token_type_ids[midindex:stopindex]
    tests_attention_masks   = attention_masks[midindex:stopindex]
    tests_number_labels     = number_labels[midindex:stopindex]
    tests_parent_labels     = parent_labels[midindex:stopindex]
    
    train_dataset = TensorDataset(train_posts_index,
                                  train_encoded_comments,
                                  train_token_type_ids,
                                  train_attention_masks,
                                  train_number_labels,
                                  train_parent_labels)
    
    tests_dataset = TensorDataset(tests_posts_index,
                                  tests_encoded_comments,
                                  tests_token_type_ids,
                                  tests_attention_masks,
                                  tests_number_labels,
                                  tests_parent_labels)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=training_batch_size,
                              shuffle=randomize)
    tests_loader = DataLoader(tests_dataset,
                              batch_size=testing_batch_size)
    
    return [train_loader, tests_loader]
    
def extract_ids(comments):
    ''' Takes a list of comments, extract the 
    comment IDs and puts into a dictionary '''
    seen_ids = set()
    for each_comment in comments:
        seen_ids.add(each_comment[''])

if __name__ =='__main__':
    time_start = time.time()
    NUM_TO_PROCESS = 1000000
    '''
    # extract the data into list of strings
    print('Flattening thread')
    pairs, _ = reddit.flatten_threads2pairs_all('coarse_discourse_dump_reddit.json');
    # filter the data with missing parents or are deleted
    print('Filtering invalid pairs')
    valid_comment_pairs = reddit.filter_valid_pairs(pairs)
    
    print('Tokenizing pairs')
    data_dict = tokenize_and_encode_pairs(valid_comment_pairs, count=NUM_TO_PROCESS)
    torch.save(data_dict, './models/test_tokenized_file.bin')
    '''
    # extract the data into list of strings
    print('Flattening threads to single level')
    comments = reddit.flatten_threads2single_all('./data/coarse_discourse_dump_reddit.json')
    filtered_comments = reddit.filter_valid_comments(comments)
    # convert to pandas dataframe
    panda_comments = reddit.comments_to_pandas(filtered_comments)
    panda_comments = reddit.pandas_remove_nan(panda_comments) 
    #data = tokenize_and_encode_pandas(panda_comments[0:NUM_TO_PROCESS])
    '''data = tokenize_and_encode_pandas(panda_comments)
    
    #data_dict = tokenize_and_encode_comments(filtered_comments, count=NUM_TO_PROCESS)
    #torch.save(data_dict, './data/first_comments_tokenized_file_with_parent_id.bin')
    torch.save(data, './data/pandas_tokenized.bin')'''
    time_end = time.time()
    time_taken = time_end - time_start
    print('Time elapsed: %6.2fs' % time_taken)
