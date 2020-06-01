#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:45:22 2020
This model tries to do pairwise classification in one shot
i.e. convert [category1:category2 ] into [category1 x 10 + category2]
@author: jakeyap
"""


import reddit_utilities as reddit
import preprocess_sentences as processor
import time
import torch
import torch.optim as optim
from classifier_models import my_ModelA1, my_ModelA2
from transformers import BertConfig

import matplotlib.pyplot as plt
import numpy as np
time_start = time.time()
    
FROM_SCRATCH = True # True if start loading model from scratch
RETOKENIZE = False # True if need to retokenize sentences again
TRAIN = False # True if you want to train the network. False to just test

'''======== FILE NAMES FLOR LOGGING ========'''
iteration = 1
MODELNAME = 'modelA3'

ITER1 = str(iteration)
DATADIR = './data/'
MODELDIR= './models/'
RESULTDIR='./results/'
# For storing the tokenized posts
posts_token_file = DATADIR + "my_tokenized_file.bin"

load_model_file = MODELDIR+MODELNAME+"_model_"+ITER1+".bin"
load_config_file = MODELDIR+MODELNAME+"_config_"+ITER1+".bin"
load_optstate_file = MODELDIR+MODELNAME+"_optimizer_"+ITER1+".bin"
load_losses_file = RESULTDIR+MODELNAME+"_losses_"+ITER1+".bin"

# Put a timestamp saved states so that overwrite accidents are less likely
timestamp = time.time()
timestamp = str("%10d" % timestamp)

ITER2 = str(iteration+1)
save_model_file = MODELDIR+MODELNAME+"_model_"+ITER2+"_"+timestamp+".bin"
save_config_file = MODELDIR+MODELNAME+"_config_"+ITER2+"_"+timestamp+".bin"
save_optstate_file = MODELDIR+MODELNAME+"_optimizer_"+ITER2+"_"+timestamp+".bin"
save_losses_file = RESULTDIR+MODELNAME+"_losses_"+ITER2+"_"+timestamp+".bin"

'''======== HYPERPARAMETERS START ========'''
NUM_TO_PROCESS = 100000
BATCH_SIZE_TRAIN = 40
BATCH_SIZE_TEST = 40
TEST_PERCENT_SPLIT = 10
LOG_INTERVAL = 10

N_EPOCHS = 1
LEARNING_RATE = 0.001
MOMENTUM = 0.5

PRINT_PICTURE = False
'''======== HYPERPARAMETERS END ========'''

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

cpu = torch.device('cpu')
gpu = torch.device('cuda')
DEVICE = cpu

# Can skip retokenizing if possible. This takes damn long ~ 10min
if RETOKENIZE:
    # extract the data into list of strings
    print('Flattening thread')
    pairs, _ = reddit.flatten_threads2pairs_all('./data/coarse_discourse_dump_reddit.json');
    # filter the data with missing parents or are deleted
    print('Filtering invalid pairs')
    valid_comment_pairs = reddit.filter_valid_pairs(pairs)
    
    print('Tokenizing pairs')
    data_dict = processor.tokenize_and_encode_pairs(valid_comment_pairs, 
                                                     count=NUM_TO_PROCESS)
    torch.save(data_dict, posts_token_file)
else:
    print('Grabbing pre-tokenized pairs')
    data_dict = torch.load(posts_token_file)

data = processor.split_dict_2_train_test_sets(data_dict=data_dict, 
                                              test_percent=TEST_PERCENT_SPLIT,
                                              training_batch_size=BATCH_SIZE_TRAIN,
                                              testing_batch_size=BATCH_SIZE_TEST,
                                              randomize=True,
                                              device=cpu)

train_loader = data[0]
tests_loader = data[1]

# count the number in the labels
pair_labels = data_dict['pair_labels']
label_counts = torch.zeros(size=(1,100), dtype=float)
for eachpair in pair_labels:
    label0 = eachpair[0]
    label1 = eachpair[1]    
    labelnum = processor.convert_label_pairs_string2num(label0, label1)     
    label_counts[0,labelnum] += 1
# add 1000 to all to make sure no division by 0 occurs 
# and for numerical stability
label_counts = label_counts + 1000
loss_sum = torch.sum(label_counts)
loss_weights = torch.true_divide(loss_sum, label_counts)
loss_weights = loss_weights.reshape(100).to('cuda')
loss_weights = torch.true_divide(loss_weights, loss_weights.max())

if FROM_SCRATCH:
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
    #                                                      num_labels=100)
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 100
    model = my_ModelA2(config)
    # Move model into GPU
    model.to(gpu)
    # Define the optimizer. Use SGD
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM)
    # Variables to store losses
    train_losses = []
    train_count = [0]
    tests_losses = []
    tests_accuracy = []
    tests_count = [0]
    
else:
    config = BertConfig.from_json_file(load_config_file)
    #model = BertForSequenceClassification(config)
    model = my_ModelA2(config)
    
    state_dict = torch.load(load_model_file)
    model.load_state_dict(state_dict)
    # Move model into GPU
    model.to(gpu)
    # Define the optimizer. Use SGD
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM)
    optim_state = torch.load(load_optstate_file)
    optimizer.load_state_dict(optim_state)
    # Variables to store losses
    losses = torch.load(load_losses_file)
    train_losses = losses[0]
    train_count = losses[1]
    tests_losses = losses[2]
    tests_accuracy = losses[3]
    tests_count = losses[4]


# Define the loss function
#loss_weights = 
loss_function = torch.nn.CrossEntropyLoss(weight=loss_weights.float(),
                                          reduction='sum')

def train(epoch):
    # Set network into training mode to enable dropout
    model.train()
    
    for batch_idx, minibatch in enumerate(train_loader):
        #move stuff to gpu
        x = minibatch[0].to(gpu)
        y = minibatch[1].to(gpu)
        token_type_ids = minibatch[2].to(gpu)
        attention_mask = minibatch[3].to(gpu)
        
        # Reset gradients to prevent accumulation
        optimizer.zero_grad()
        # Forward prop throught BERT
        outputs = model(input_ids = x,
                        attention_mask=attention_mask, 
                        token_type_ids=token_type_ids)
        
        #outputs is a length=1 tuple. Get index 0 to access real outputs
        # Calculate loss
        loss = loss_function(outputs[0], y)
        # Backward prop find gradients
        loss.backward()
        # Update weights & biases
        optimizer.step()
        
        #delete references to free up GPU space
        del x, y, token_type_ids, attention_mask
        
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * BATCH_SIZE_TRAIN, len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), 
                  loss.item() / BATCH_SIZE_TRAIN))
            
            train_losses.append(loss.item() / BATCH_SIZE_TRAIN)
            train_count.append(train_count[-1] + BATCH_SIZE_TEST*LOG_INTERVAL)
            
            # Store the states of model and optimizer into logfiles
            # In case training gets interrupted, you can load old states
            
            torch.save(model.state_dict(), save_model_file)
            torch.save(optimizer.state_dict(), save_optstate_file)
            torch.save([train_losses,
                        train_count,
                        tests_losses,
                        tests_accuracy,
                        tests_count], save_losses_file)
            model.config.to_json_file(save_config_file)

def test(save=False):
    # This function evaluates the entire test set
    
    # Set network into evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    # start the label arrays. 1st data point has to be deleted later
    predicted_label_arr = torch.tensor([[0]])
    groundtruth_arr = torch.tensor([0])
    #predicted_label_arr = torch.zeros(1, len(tests_loader.dataset))
    with torch.no_grad():
        for batchid, minibatch in enumerate(tests_loader):
            x = minibatch[0].to('cuda')
            y = minibatch[1].to('cuda')
            token_type_ids = minibatch[2].to('cuda')
            attention_mask = minibatch[3].to('cuda')
            outputs = model(input_ids = x,
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)
            #outputs is a length=1 tuple. Get index 0 to access real outputs
            outputs = outputs[0]
            test_loss += loss_function(outputs, y).item()
            
            predicted_label = outputs.data.max(1, keepdim=True)[1]
            correct += predicted_label.eq(y.data.view_as(predicted_label)).sum()
            
            #predicted_label_list.append(predicted_label.to('cpu'))
            predicted_label_arr = torch.cat((predicted_label_arr,
                                             predicted_label.to('cpu')),
                                            0)
            groundtruth_arr = torch.cat((groundtruth_arr,
                                         y.to('cpu')),
                                        0)
            #delete references to free up GPU space
            del x, y, token_type_ids, attention_mask
    test_loss /= len(tests_loader.dataset)
    if save:
        tests_losses.append(test_loss)
        tests_accuracy.append(100. * correct.to('cpu') / len(tests_loader.dataset))
        tests_count.append(tests_count[-1] + len(train_loader.dataset))
        torch.save([train_losses,
                    train_count,
                    tests_losses,
                    tests_accuracy,
                    tests_count], 
                   save_losses_file)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(tests_loader.dataset),
          100. * correct / len(tests_loader.dataset)))
    return predicted_label_arr[1:], groundtruth_arr[1:]


def eval_single_example(number_to_check, show=True):
    batch_to_check = number_to_check // BATCH_SIZE_TEST
    index_to_check = number_to_check % BATCH_SIZE_TEST
    
    with torch.no_grad():
        for batchid, minibatch in enumerate(tests_loader):
            if batchid == batch_to_check:
                x = minibatch[0].to('cuda')
                y = minibatch[1].to('cuda')
                token_type_ids = minibatch[2].to('cuda')
                attention_mask = minibatch[3].to('cuda')
                outputs = model(input_ids = x,
                                attention_mask=attention_mask, 
                                token_type_ids=token_type_ids)
                
                outputs = outputs[0]
                # classification is the one with highest score output score
                prediction = outputs.argmax(dim=1)
                prediction = processor.convert_label_pairs_num2string(prediction[index_to_check])
                encoded_sentence = x[index_to_check,0:]
                reallabels = processor.convert_label_pairs_num2string(y[index_to_check])
                
                if show:
                    print('Original sentences:')
                    print(reddit.tokenizer.decode(encoded_sentence.tolist()))
                    print('Original Labels: \t', reallabels[0], '\t', reallabels[1])
                    print('Predicted Labels: \t', prediction[0], '\t', prediction[1])
                
                del x, y, token_type_ids, attention_mask, outputs
                del encoded_sentence, 
                return reallabels, prediction

def plot_losses(offset=5):
    fig = plt.figure(1)
    try:
        losses = torch.load(save_losses_file)
    except Exception:
        losses = torch.load(load_losses_file)
    train_losses = losses[0]
    train_count = losses[1]
    tests_losses = losses[2]
    tests_accuracy = losses[3]
    tests_count = losses[4]
    
    plt.scatter(train_count[1+offset:], 
                train_losses[offset:], label='Train')
    plt.scatter(tests_count[1:], tests_losses, label='Test')
    plt.ylabel('Loss')
    plt.xlabel('Minibatches seen. Batchsize='+str(BATCH_SIZE_TEST))
    plt.legend(loc='best')
    plt.grid(True)
    return losses

if __name__ =='__main__':
    test(save=False)
    if (TRAIN):
        for epoch in range(1, N_EPOCHS + 1):
            train(epoch)
            labels = test(save=True)
    plot_losses()

    time_end = time.time()
    time_taken = time_end - time_start
    print('Time elapsed: %6.2fs' % time_taken)