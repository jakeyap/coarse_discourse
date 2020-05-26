# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:11:50 2020

@author: Yong Keong
"""
import json 
import reddit_utilities as reddit
import time

import numpy as np
import matplotlib.pyplot as plt
import main
from preprocess_sentences import convert_label_pairs_num2string

# Grab the true labels
pair_labels = main.data_dict['pair_labels']
label_counts = np.zeros(shape=(10,10), dtype=int)
test_label_counts = np.zeros(shape=(10,10), dtype=int)

label_dict = reddit.empty_label_dictionary()
label_list = list(label_dict.keys())

for eachpair in pair_labels:
    label0 = eachpair[0]
    label1 = eachpair[1]
    
    index0 = label_list.index(label0)
    index1 = label_list.index(label1)
    
    label_counts[index0, index1] += 1
    
    
# Grab the predicted labels
predicted = main.test(save=False)
# Convert into a row
predicted = predicted.reshape(-1)
test_len = predicted.shape[0]
counter = 0
while counter < test_len:
    test_pair_labels = convert_label_pairs_num2string(predicted[counter])
    label0 = test_pair_labels[0]
    label1 = test_pair_labels[1]
    index0 = label_list.index(label0)
    index1 = label_list.index(label1)
    test_label_counts[index0, index1] += 1
    counter += 1


plt.figure(1)
plt.title('True labels')
plt.imshow(label_counts, cmap='gray')
ax = plt.gca()
# We want to show all ticks...
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))

# ... and label them with the respective list entries
ax.set_xticklabels(label_list)
ax.set_yticklabels(label_list)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(10):
    for j in range(10):
        number = label_counts[i, j]
        if number > 20000:
            text = ax.text(j, i, str(number),
                           ha="center", va="center", color="black")
        else:
            text = ax.text(j, i, str(number),
                           ha="center", va="center", color="white")

plt.colorbar()
plt.ylabel('1st comment', size=10)
plt.xlabel('2nd comment', size=10)
#imshow(label_count)
#plt.colorbar()

plt.figure(2)
plt.title('Test predictions')
plt.imshow(test_label_counts, cmap='gray')
ax2 = plt.gca()
# We want to show all ticks...
ax2.set_xticks(np.arange(10))
ax2.set_yticks(np.arange(10))

# ... and label them with the respective list entries
ax2.set_xticklabels(label_list)
ax2.set_yticklabels(label_list)

# Rotate the tick labels and set their alignment.
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(10):
    for j in range(10):
        number = test_label_counts[i, j]
        if number > 2000:
            text = ax2.text(j, i, str(number),
                            ha="center", va="center", color="black")
        else:
            text = ax2.text(j, i, str(number),
                            ha="center", va="center", color="white")
            
plt.colorbar()
plt.ylabel('1st comment', size=10)
plt.xlabel('2nd comment', size=10)