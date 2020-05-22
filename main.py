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
from classifier_models import my_BERT_Model
from transformers import BertForSequenceClassification, BertConfig
time_start = time.time()


'''======== FILE NAMES FLOR LOGGING ========'''
iteration = 0

load_output_model_file = "./models/my_own_model_file"+str(iteration)+".bin"
load_output_config_file = "./models/my_own_config_file"+str(iteration)+".bin"
load_optim_state_file = "./models/my_optimizer_file"+str(iteration)+".bin"
#load_output_vocab_file = "./models/my_own_vocab_file"+str(iteration)+".bin"

save_output_model_file = "./models/my_own_model_file"+str(iteration+1)+".bin"
save_output_config_file = "./models/my_own_config_file"+str(iteration+1)+".bin"
save_optim_state_file = "./models/my_optimizer_file"+str(iteration+1)+".bin"
#save_output_vocab_file = "./models/my_own_vocab_file"+str(iteration+1)+".bin"

FROM_SCRATCH = True # True if start loading model from scratch

'''======== HYPERPARAMETERS START ========'''
NUM_TO_PROCESS = 1000
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
TEST_PERCENT_SPLIT = 10
LOG_INTERVAL = 10

N_EPOCHS = 3
LEARNING_RATE = 0.001
MOMENTUM = 0.1

PRINT_PICTURE = False
'''======== HYPERPARAMETERS END ========'''

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

cpu = torch.device('cpu')
gpu = torch.device('cuda')
DEVICE = cpu

# extract the data into list of strings
pairs, _ = reddit.flatten_threads2pairs_all('coarse_discourse_dump_reddit.json');
# filter the data with missing parents or are deleted
valid_comment_pairs = reddit.filter_valid_pairs(pairs)


data_dict = processor.tokenize_and_encode_pairs(valid_comment_pairs, 
                                                 count=NUM_TO_PROCESS)

data = processor.split_dict_2_train_test_sets(data_dict=data_dict, 
                                              test_percent=TEST_PERCENT_SPLIT,
                                              training_batch_size=BATCH_SIZE_TRAIN,
                                              testing_batch_size=BATCH_SIZE_TEST,
                                              randomize=False,
                                              device=cpu)

train_loader = data[0]
tests_loader = data[1]
train_examples = enumerate(train_loader)
tests_examples = enumerate(tests_loader)

'''
print('Show 1 example of training examples')
batch_id, minibatch1 = next(tests_examples)
x = minibatch1[0].to(gpu)
print(x)
print('x shape is: ', x.shape)
y = minibatch1[1].to(gpu)
token_type_ids = minibatch1[2].to(gpu)
attention_mask = minibatch1[3].to(gpu)
'''


if FROM_SCRATCH:
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
    #                                                      num_labels=100)
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 100
    model = my_BERT_Model(config)
    # Define the optimizer. Use SGD
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM)
else:
    config = BertConfig.from_json_file(load_output_config_file)
    #model = BertForSequenceClassification(config)
    model = my_BERT_Model(config)
    
    state_dict = torch.load(load_output_model_file)
    model.load_state_dict(state_dict)
    
    # Define the optimizer. Use SGD
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM)
    optim_state = torch.load(load_optim_state_file)
    optimizer.load_state_dict(optim_state)


# Turn off drop out randomizer
model.eval()
# Move model into GPU
model.to(gpu)

# Define the loss function
loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
#loss_fn = MSELoss(reduction='sum') then test_loss += loss_fn(output, label).item()

# Variables to store losses
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(N_EPOCHS + 1)]


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
        '''
        x = x.to(cpu)
        y = x.to(cpu)
        token_type_ids = train_examples[2].to(cpu)
        attention_mask = train_examples[3].to(cpu)
        '''
        del x, y, token_type_ids, attention_mask
        
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * BATCH_SIZE_TRAIN, len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))
            
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*BATCH_SIZE_TRAIN) + ((epoch-1)*len(train_loader.dataset))
            )
            
            # Store the states of model and optimizer into logfiles
            # In case training gets interrupted, you can load old states
            
            torch.save(model.state_dict(), save_output_model_file)
            torch.save(optimizer.state_dict(), save_optim_state_file)
            model.config.to_json_file(save_output_config_file)
            #tokenizer.save_vocabulary(output_vocab_file)
            

def test():
    # This function evaluates the entire test set
    
    # Set network into evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
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
            
            #delete references to free up GPU space
            del x, y, token_type_ids, attention_mask
            
    test_loss /= len(tests_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(tests_loader.dataset),
          100. * correct / len(tests_loader.dataset)))

test()
for epoch in range(1, N_EPOCHS + 1):
    train(epoch)
    test()

#TODO
    # find an example pair's labels
    # check the example labels
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
            '''
            '''

'''   
# The loss inside the model is automatically selected to be the 
# multi-class NLL loss.
out = model(input_ids = x,
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids)

'''
time_end = time.time()
time_taken = time_end - time_start
print('Time elapsed: %6.2fs' % time_taken)