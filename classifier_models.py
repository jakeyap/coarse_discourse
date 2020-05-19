#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:58:11 2020

@author: jakeyap
"""
import torch
from transformers import BertTokenizer, BertModel


categories = ['question',
              'answer',
              'announcement',
              'agreement',
              'appreciation',
              'disagreement',
              'negativereaction',
              'elaboration',
              'humor',
              'other',]

#TODO: Create a function to convert categories into one hot vectors

#TODO: To add models here

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

json_filename = "coarse_discourse_dump_reddit.json"

