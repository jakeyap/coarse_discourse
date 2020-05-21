#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:45:22 2020

@author: jakeyap
"""


import reddit_utilities as reddit
import preprocess_sentences as processor
import time
import torch

from transformers import BertModel, BertForSequenceClassification

time_start = time.time()
''' Hyperparameters start '''

NUM_TO_PROCESS = 1000
training_batch_size = 64
testing_batch_size = 64
TEST_PERCENT = 10
''' Hyperparameters end'''

# extract the data into list of strings
pairs, _ = reddit.flatten_threads2pairs_all('coarse_discourse_dump_reddit.json');
# filter the data with missing parents or are deleted
valid_comment_pairs = reddit.filter_valid_pairs(pairs)


data_dict = processor.tokenize_and_encode_pairs(valid_comment_pairs, 
                                                 count=NUM_TO_PROCESS)

data = processor.split_dict_2_train_test_sets(data_dict=data_dict, 
                                              test_percent=TEST_PERCENT,
                                              training_batch_size=16,
                                              testing_batch_size=16,
                                              randomize=False,
                                              device='cpu')

train_loader = data[0]
tests_loader = data[1]
train_examples = enumerate(train_loader)
tests_examples = enumerate(tests_loader)

print('Show 1 example of training examples')
batch_id, train_example = next(train_examples)
x = train_example[0]
print(x)
print('x shape is: ', x.shape)
#y = train_example[1].to('cuda')
#token_type_ids = train_example[2].to('cuda')
#attention_mask = train_example[3].to('cuda')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.config.num_labels = 100
# Turn off drop out randomizer
model.eval()
# Move model into GPU
model.to('cuda')

# Define the loss function
loss_function = torch.nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE,
                      momentum=MOMENTUM)

def train(epoch):
    # Set network into training mode to enable dropout
    model.train()
    
    for batch_idx, train_examples in enumerate(train_loader):
        x = train_example[0].to('cuda')
        y = train_example[1].to('cuda')
        token_type_ids = train_example[2].to('cuda')
        attention_mask = train_example[3].to('cuda')
        
        # Reset gradients to prevent accumulation
        optimizer.zero_grad()
        # Forward prop
        predictions = model(data)
        # Calculate loss
        loss = loss_function(predictions, y)
        # Backward prop find gradients
        loss.backward()
        # Update weights & biases
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))
            
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset))
            )
            
            # Store the states of model and optimizer into logfiles
            # In case training gets interrupted, you can load old states
            torch.save(model.state_dict(), './results/model_state.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')



# The loss inside the model is automatically selected to be the 
# multi-class NLL loss.
out = model(input_ids = x,
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids)


time_end = time.time()
time_taken = time_end - time_start
print('Time elapsed: %6.2fs' % time_taken)