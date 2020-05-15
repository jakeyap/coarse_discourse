#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:18:44 2020

@author: Yong Keong
"""


import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
print('Loading pre-trained model tokenizer')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
# CLS means classification. SEP means sentence separator
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
text2 = "[CLS] Where did he go ? [SEP] He went to the office ! [SEP]"

print("text is: ", text)
print("text2 is: ", text2)
print('Tokenize input')
tokenized_text = tokenizer.tokenize(text)
tokenized_text2 = tokenizer.tokenize(text2)

# Mask a token that we will try to predict back with `BertForMaskedLM`
print("Masking sentences: ")
masked_index = 1
tokenized_text[masked_index] = '[MASK]'
tokenized_text2[masked_index] = '[MASK]'
print(list(tokenized_text))
print(list(tokenized_text2))
#assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
segments_ids2 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
tokens_tensor2 = torch.tensor([indexed_tokens2])

segments_tensors = torch.tensor([segments_ids])
segments_tensors2 = torch.tensor([segments_ids2])

print("==== This part is for encoding an input into hidden states ====")

# Load pre-trained model (weights)
print("loading pre-trained model")
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
print("Disabling dropout for deployment")
model.eval()

# If you have a GPU, put everything on cuda
print("Setting calculations to GPU")
tokens_tensor = tokens_tensor.to('cuda')
tokens_tensor2 = tokens_tensor2.to('cuda')
segments_tensors = segments_tensors.to('cuda')
segments_tensors2 = segments_tensors2.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
print("Predicting hidden states for each layer")
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    # Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer of the Bert model
    encoded_layers = outputs[0]
# We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)

print("==== This part is for predicting masked sentences ====")
# Load pre-trained model (weights)
print("load pre-trained model")
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# If you have a GPU, put everything on cuda
print("Setting calculations to GPU")
tokens_tensor = tokens_tensor.to('cuda')
tokens_tensor2 = tokens_tensor2.to('cuda')
segments_tensors = segments_tensors.to('cuda')
segments_tensors2 = segments_tensors2.to('cuda')
model.to('cuda')

# Predict all tokens
print("Predicting all tokens")
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    outputs2 = model(tokens_tensor2, token_type_ids=segments_tensors2)
    predictions = outputs[0]
    predictions2 = outputs2[0]

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
#assert predicted_token == 'henson'

predicted_index2 = torch.argmax(predictions2[0, masked_index]).item()
predicted_token2 = tokenizer.convert_ids_to_tokens([predicted_index2])[0]
print('1st predicted word: ', predicted_token)
print('2nd predicted word: ', predicted_token2)