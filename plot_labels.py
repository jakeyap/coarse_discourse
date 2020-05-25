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

pair_labels = data_dict['pair_labels']
label_counts = np.zeros(shape=(10,10), dtype=int)

label_dict = reddit.empty_label_dictionary()
label_list = list(label_dict.keys())

for eachpair in pair_labels:
    label0 = eachpair[0]
    label1 = eachpair[1]
    
    index0 = label_list.index(label0)
    index1 = label_list.index(label1)
    
    label_counts[index0, index1] += 1
    
fig, ax = plt.subplots()
plt.imshow(label_counts, cmap='gray')
ax = plt.gca()
# We want to show all ticks...
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))

# ... and label them with the respective list entries
ax.set_xticklabels(label_list,size=8)
ax.set_yticklabels(label_list, size=8)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, 
         ha="right",rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(10):
    for j in range(10):
        number = label_counts[i, j]
        if number > 25000:
            text = ax.text(j, i, str(number),
                           ha="center", va="center", 
                           color="black", size=8)
        else:
            text = ax.text(j, i, str(number),
                           ha="center", va="center", 
                           color="white", size=8)

plt.colorbar(aspect=50)
plt.title('Pairwise comment types', size=15)
plt.ylabel('1st comment type', size=10)
plt.xlabel('2nd comment type', size=10)

#plt.tight_layout()
#imshow(label_count)
#plt.colorbar()
