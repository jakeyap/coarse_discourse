#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:37:32 2020

@author: jakeyap
"""

import reddit_utilities as reddit
import preprocess_sentences as processor
import time
import torch
import torch.optim as optim
from classifier_models import my_ModelB1
from transformers import BertConfig
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import numpy as np
time_start = time.time()
    
FROM_SCRATCH = False # True if start loading model from scratch
TRAIN = False # True if you want to train the network. False to just test
SEQUENTIAL_PREDICTION = True # True if want to predict head's label then predict child

'''======== FILE NAMES FOR LOGGING ========'''
iteration = 1
MODELNAME = 'modelB1'

ITER1 = str(iteration)
DATADIR = './data/'
MODELDIR= './models/MODELB/'
RESULTDIR='./results/MODELB/'
# For storing the tokenized posts
posts_token_file = DATADIR + "pandas_tokenized.bin"
#posts_token_file = DATADIR + "first_comments_tokenized_file.bin"

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

N_EPOCHS = 10
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

print('Importing original tokenized & encoded comments')
dataframe = torch.load(posts_token_file)

print('Splitting into training and test sets')
data = processor.split_pandas_2_train_test_sets(pandas_dataframe=dataframe, 
                                                test_percent=TEST_PERCENT_SPLIT,
                                                training_batch_size=BATCH_SIZE_TRAIN,
                                                testing_batch_size=BATCH_SIZE_TEST,
                                                randomize=True,
                                                split_first_post=SEQUENTIAL_PREDICTION,
                                                DEBUG=False)

train_loader = data[0]
tests_loader = data[1]
if SEQUENTIAL_PREDICTION:
    first_tests_loader = data[2]
'''
# count the number in the labels
labels = dataframe['labels']
label_counts = torch.zeros(size=(1,100), dtype=float)
for each_label in labels:
    labelnum = processor.convert_label_string2num(each_label)
    label_counts[0,labelnum] += 1
'''
'''
# add 1000 to all to make sure no division by 0 occurs 
# and for numerical stability
label_counts = label_counts + 1000
loss_sum = torch.sum(label_counts)
loss_weights = torch.true_divide(loss_sum, label_counts)
loss_weights = loss_weights.reshape(100).to('cuda')
loss_weights = torch.true_divide(loss_weights, loss_weights.max())
'''

if FROM_SCRATCH:
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
    #                                                      num_labels=100)
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 10
    model = my_ModelB1(config)
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
    model = my_ModelB1(config)
    
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
'''
loss_function = torch.nn.CrossEntropyLoss(weight=loss_weights.float(),
                                          reduction='sum')
'''
loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

def train(epoch):
    # Set network into training mode to enable dropout
    model.train()

    train_loader = data[0]
    
    for batch_idx, minibatch in enumerate(train_loader):
        '''
        index in original data,
        x (encoded comment), 
        token_typed_ids,
        attention_masks,
        y (true label),
        parent_y (parent's label)
        '''
        #move stuff to gpu
        x = minibatch[1].to(gpu)
        token_type_ids = minibatch[2].to(gpu)
        attention_mask = minibatch[3].to(gpu)
        own_y = minibatch[4].to(gpu)
        length = own_y.shape[0]
        own_y = own_y.reshape(length)
        parent_y = minibatch[5].float().to(gpu)
        
        # Reset gradients to prevent accumulation
        optimizer.zero_grad()
        # Forward prop throught BERT
        outputs = model(input_ids = x,
                        attention_mask=attention_mask, 
                        token_type_ids=token_type_ids,
                        parent_labels =parent_y)
        
        #outputs is a length=1 tuple. Get index 0 to access real outputs
        # Calculate loss
        loss = loss_function(outputs[0], own_y)
        # Backward prop find gradients
        loss.backward()
        # Update weights & biases
        optimizer.step()
        
        #delete references to free up GPU space
        del x, token_type_ids, attention_mask, own_y, parent_y
        
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {:2d} [{:5d}/{:5d} ({:2.1f}%)]\tLoss: {:2.4f}'.format(
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
    
    with torch.no_grad():
        for batchid, minibatch in enumerate(tests_loader):
            x = minibatch[1].to(gpu)
            token_type_ids = minibatch[2].to(gpu)
            attention_mask = minibatch[3].to(gpu)
            own_y = minibatch[4].to(gpu)
            length = own_y.shape[0]
            own_y = own_y.reshape(length)
            parent_y = minibatch[5].float().to(gpu)
            
            outputs = model(input_ids = x,
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids,
                            parent_labels =parent_y)
            #outputs is a length=1 tuple. Get index 0 to access real outputs
            outputs = outputs[0]
            test_loss += loss_function(outputs, own_y).item()
            
            predicted_label = outputs.data.max(1, keepdim=True)[1]
            correct += predicted_label.eq(own_y.data.view_as(predicted_label)).sum()
            
            predicted_label_arr = torch.cat((predicted_label_arr,
                                             predicted_label.to(cpu)),
                                            0)
            groundtruth_arr = torch.cat((groundtruth_arr,
                                         own_y.to(cpu)),
                                        0)
            #delete references to free up GPU space
            del x, token_type_ids, attention_mask, own_y, parent_y
    test_loss /= len(tests_loader.dataset)
    if save:
        tests_losses.append(test_loss)
        tests_accuracy.append(100. * correct.to(cpu) / len(tests_loader.dataset))
        tests_count.append(tests_count[-1] + len(train_loader.dataset))
        torch.save([train_losses,
                    train_count,
                    tests_losses,
                    tests_accuracy,
                    tests_count], 
                   save_losses_file)
    
    predicted_label_arr = predicted_label_arr.reshape(shape=(-1,))
    groundtruth_arr = groundtruth_arr.reshape(shape=(-1,))
    score = f1_score(groundtruth_arr, predicted_label_arr, average='micro')
    total_len = len(tests_loader.dataset)
    print('\nTest set: Avg loss: {:3.4f}'.format(test_loss), end='\t')
    print(' Accuracy: {:6d}/{:6d} ({:2.1f}%)'.format(correct, total_len, 100. * correct / total_len), end='\t')
    print(' F1 score: {:1.2f}'.format(score))
    return predicted_label_arr[1:], groundtruth_arr[1:]

def eval_single_example(number_to_check=None, show=True):
    datalength = len(tests_loader.dataset)
    if number_to_check is None:
        # Roll a random number
        number_to_check = torch.randint(low=0, high=datalength,size=(1,1))
        # Use this number as the index
        number_to_check = number_to_check.item()
    batch_to_check = number_to_check // BATCH_SIZE_TEST
    index_to_check = number_to_check % BATCH_SIZE_TEST
    
    with torch.no_grad():
        for batchid, minibatch in enumerate(tests_loader):
            if batchid == batch_to_check:
                x = minibatch[1].to(gpu)
                token_type_ids = minibatch[2].to(gpu)
                attention_mask = minibatch[3].to(gpu)
                own_y = minibatch[4].to(gpu)
                length = own_y.shape[0]
                own_y = own_y.reshape(length)
                parent_y = minibatch[5].float().to(gpu)
                outputs = model(input_ids = x,
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids,
                            parent_labels =parent_y)
                
                outputs = outputs[0]
                # classification is the one with highest score output score
                
                prediction = outputs.argmax(dim=1)
                prediction = processor.convert_label_num2string(prediction[index_to_check])
                comment = x[index_to_check,0:]
                reallabel = processor.convert_label_num2string(own_y[index_to_check])
                parentlabel = processor.convert_label_num2string(parent_y[index_to_check].int())
                if show:
                    print('Original sentences:')
                    print(processor.tokenizer.decode(comment.tolist()))
                    print('Original Labels: \t', reallabel)
                    print('Predicted Labels: \t', prediction)
                    print('Parent label: \t\t', parentlabel)
                
                del x, token_type_ids, attention_mask, own_y, length, parent_y
                del outputs, comment
                return reallabel, prediction, parentlabel

def test_sequential(save=False):
    # This function evaluates the entire test set.  
    # For the test set, we are going to evaluate parent first 
    # before evaluating self
    
    # Set network into evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    # start the label arrays. 1st data point has to be deleted later
    predicted_label_arr = torch.tensor([[0]])
    groundtruth_arr = torch.tensor([0])
    # test the main batch
    with torch.no_grad():
        for batchid, minibatch in enumerate(tests_loader):
            # get the index of this sample in the database
            sample_index = minibatch[0].reshape(-1)
            # get the parent ids from the original database
            parent_ids   = dataframe.iloc[sample_index]['majority_link'].values
            # find the parent posts' index
            parents_index = find_parents_index(parent_ids)
            
            parent_info = find_posts_info(parents_index)
            parent_encoded_comments = parent_info['encoded_comments'].to(gpu)
            parent_token_ids = parent_info['token_type_ids'].to(gpu)
            parent_attention = parent_info['attention_masks'].to(gpu)
            grandparent_label= parent_info['parent_label'].to(gpu)
            
            #print(type(grandparent_label))
            # generate parents' label by running them thru model
            parent_outputs = model(input_ids = parent_encoded_comments,
                                   attention_mask=parent_attention, 
                                   token_type_ids=parent_token_ids,
                                   parent_labels =grandparent_label)
            
            # get the generated parent labels
            parent_outputs = parent_outputs[0]
            generated_parent_label = parent_outputs.data.max(1, keepdim=True)[1]
            generated_parent_label = generated_parent_label.float()
            
            x = minibatch[1].to(gpu)
            token_type_ids = minibatch[2].to(gpu)
            attention_mask = minibatch[3].to(gpu)
            own_y = minibatch[4].to(gpu)
            length = own_y.shape[0]
            own_y = own_y.reshape(length)
            #parent_y = minibatch[5].float().to(gpu)
            
            outputs = model(input_ids = x,
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids,
                            parent_labels =generated_parent_label)
            #outputs is a length=1 tuple. Get index 0 to access real outputs
            outputs = outputs[0]
            test_loss += loss_function(outputs, own_y).item()
            
            predicted_label = outputs.data.max(1, keepdim=True)[1]
            correct += predicted_label.eq(own_y.data.view_as(predicted_label)).sum()
            
            predicted_label_arr = torch.cat((predicted_label_arr,
                                             predicted_label.to(cpu)),
                                            0)
            groundtruth_arr = torch.cat((groundtruth_arr,
                                         own_y.to(cpu)),
                                        0)
            #delete references to free up GPU space
            #del x, token_type_ids, attention_mask, own_y, parent_y
            del x, token_type_ids, attention_mask, own_y
            
        # for the first posts, dont need to do anything fancy. 
        # just plug in directly
        for batch_id, minibatch in enumerate(first_tests_loader):
            x = minibatch[1].to(gpu)
            token_type_ids = minibatch[2].to(gpu)
            attention_mask = minibatch[3].to(gpu)
            own_y = minibatch[4].to(gpu)
            length = own_y.shape[0]
            own_y = own_y.reshape(length)
            parent_y = minibatch[5].float().to(gpu)
            
            outputs = model(input_ids = x,
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids,
                            parent_labels =parent_y)
            #outputs is a length=1 tuple. Get index 0 to access real outputs
            outputs = outputs[0]
            test_loss += loss_function(outputs, own_y).item()
            
            predicted_label = outputs.data.max(1, keepdim=True)[1]
            correct += predicted_label.eq(own_y.data.view_as(predicted_label)).sum()
            
            predicted_label_arr = torch.cat((predicted_label_arr,
                                             predicted_label.to(cpu)),
                                            0)
            groundtruth_arr = torch.cat((groundtruth_arr,
                                         own_y.to(cpu)),
                                        0)
            #delete references to free up GPU space
    test_size = len(tests_loader.dataset) + len(first_tests_loader.dataset)
    test_loss /= test_size
    if save:
        tests_losses.append(test_loss)
        tests_accuracy.append(100. * correct.to(cpu) / test_size)
        tests_count.append(tests_count[-1] + len(train_loader.dataset))
        torch.save([train_losses,
                    train_count,
                    tests_losses,
                    tests_accuracy,
                    tests_count], 
                   save_losses_file)
    
    predicted_label_arr = predicted_label_arr.reshape(shape=(-1,))
    groundtruth_arr = groundtruth_arr.reshape(shape=(-1,))
    score = f1_score(groundtruth_arr, predicted_label_arr, average='micro')
    print('\nTest set: Avg loss: {:3.4f}'.format(test_loss), end='\t')
    print(' Accuracy: {:6d}/{:6d} ({:2.1f}%)'.format(correct, test_size, 100. * correct / test_size), end='\t')
    print(' F1 score: {:1.2f}'.format(score))
    return predicted_label_arr[1:], groundtruth_arr[1:]

def plot_losses(offset=0):
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

def find_parent_index(string_id):
    ''' Given a parent ID string, find parent index'''
    counter = 0
    row = dataframe[dataframe['id']==string_id]
    row_number = row.index.item()
    return row_number
''' 
Try this
dataframe[dataframe.iloc['id']=='t3_1kun69']
You get 2 parents!!?!?
'''
def find_parents_index(parent_ids):
    ''' Given a numpy array with parent ID string, find parent indices '''
    parents_index = torch.ones(len(parent_ids)) * -1
    i = 0
    while i < len(parent_ids):
        string_id = parent_ids[i]
        row = dataframe[dataframe['id']==string_id]
        #print(row)
        print(row)
        row_number = row.index.item()
        parents_index[i] = int(row_number)
        i += 1
    return parents_index

def find_posts_info(indices):
    ''' Given an array of indices, spit out the 
    encoded_comments, token_type_id, attention_masks, parent_label
    '''
    # find posts encoded data
    encoded_comments = dataframe.iloc[indices]['encoded_comments'].values
    encoded_comments = np.array(encoded_comments.tolist())
    encoded_comments = torch.tensor(encoded_comments)
    # find parent posts token_type_ids
    token_type_ids  = dataframe.iloc[indices]['token_type_ids'].values
    token_type_ids  = np.array(token_type_ids.tolist())
    token_type_ids  = torch.tensor(token_type_ids)
    # find parent posts attention mask
    attention_masks = dataframe.iloc[indices]['attention_masks'].values
    attention_masks = np.array(attention_masks.tolist())
    attention_masks = torch.tensor(attention_masks)
    # find parent posts' parent labels
    parent_label    = dataframe.iloc[indices]['parent_labels'].values
    parent_label    = torch.tensor(parent_label).float().reshape((-1,1))
    
    return {'encoded_comments': encoded_comments,
            'token_type_ids'  : token_type_ids, 
            'attention_masks' : attention_masks,
            'parent_label'    : parent_label}

if __name__ =='__main__':
    if SEQUENTIAL_PREDICTION:
        test_sequential(save=False)
    else:
        test(save=False)
    if (TRAIN):
        for epoch in range(1, N_EPOCHS + 1):
            train(epoch)
            labels = test(save=True)
    plot_losses()

    time_end = time.time()
    time_taken = time_end - time_start
    print('Time elapsed: %6.2fs' % time_taken)