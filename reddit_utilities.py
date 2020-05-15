#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:16:05 2020
Some utility functions to help with data processing
@author: jakeyap
"""
from transformers import BertTokenizer
print('Loading pre-trained model tokenizer')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def flatten_posts(thread_json, comment_pairs=[]):
    ''' 
    Arguments: 
        thread_json:    a reddit thread in json format
        comment_pairs:  list of lists (see output)
    Output:
        comment_pairs:  list of lists
            [
                [comment1 comment1.1]
                [comment1.1 comment1.1.1]
                [comment1.1 comment1.1.2]
            ]
    Notes: 
        1. the tree will be traversed in a depth first order
        2. if the comment has no body, it will be discarded
    '''
    
    # posts_json is a list of dicts
    posts_json = thread_json['posts']
    
    # lookuptable is just a list that all the post ids. 
    # For quickly checking what parent a post links to    
    lookuptable = []
    
    for each_post in posts_json:
        # Go through whole thread once, 
        # Store all post_ids in a list
        lookuptable.append(each_post['id'])
    
    for each_post in posts_json:    
        if post_is_first(each_post):
            # Skip cauz cant generate pairs with just 1st post
            pass
        
        elif not post_has_body(each_post):
            # Skip cauz the post has no body
            pass
        
        else:
            try:
                # Find current post's parent
                parent = post_parent(each_post, posts_json, lookuptable)
                # Store [parent, child] into the global list of lists
                comment_pairs.append([parent, each_post])
            except Exception:
                # sometimes, just cannot find parent post. skip
                pass
    return comment_pairs

'''
Some helper functions below
'''

def post_is_first(post_json):
    ''' 
    Determine if post is the first in thread 
    Check whether the json field 'is_first_post' exists
    '''
    if 'is_first_post' in post_json.keys():
        return True
    else:
        return False
    
def post_has_body(post_json):
    ''' 
    Determine if post has text/body. If no, data is missing
    Check whether the json field 'body' exists
    '''
    if 'body' in post_json.keys():
        return True
    else:
        return False    
    
def post_parent(single_post_json, posts_json, lut):
    '''
    Args:       single_post_json (a post in json format)
                posts_json all the posts in the current thread
                lut (a list of post IDs)
    Returns:    parent post
    
    Just look at the 'in_reply_to' json field
    In some cases, the 'in_reply_to' is linked to a bot post
    In those cases, use 'majority_link' instead
    Other times, the comment is deleted from database. 
    '''
    
    try:
        # Find ID of current post's parent
        parent_id = single_post_json['in_reply_to']
        
        # Look up that ID's position in the json thread
        parent_index = lut.index(parent_id)
        
        # Find that parent
        parent = posts_json[parent_index]
        
    except Exception:        
        # Find ID of current post's parent
        parent_id = single_post_json['majority_link']
        
        # Look up that ID's position in the json thread
        parent_index = lut.index(parent_id)
        '''
        print(type(lut), lut)
        print(parent_index)
        print(type(posts_json))
        print(post_json)
        '''
        # Find that parent
        parent = posts_json[parent_index]
    # If trying majority link still doesnt give a parent
    # Just give up on this post. Throw error
    return parent

#TODO
def count_sentence(post):
    '''
    Counts number of sentences in a post
    '''
    return


#TODO
def count_words(sentence):
    '''
    Counts number of words in a sentence
    '''
    print("text is: ", text)
    print('Tokenize input')
    tokenized_text = tokenizer.tokenize(text)