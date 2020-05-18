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

import numpy as np

def flatten_threads2pairs_all(filename, comments=[], start=0, count=100000):
    """
    Flattens entire reddit json file into comment pairs.
    1. the tree will be traversed in a depth first order
    2. if the comment has no body, it will be discarded
    
    Parameters
    ----------
    filename : string 
        Name of a .json file
    comments : list of lists, optional
        See output. The default is [].
    start : int, optional
        Index to start from. The default is 0.
    count : int, optional
        Number of threads to do. The default is 100000.

    Returns
    -------
    comments : list of lists
        [
            [comment1 comment1.1]
            [comment1.1 comment1.1.1]
            [comment1.1 comment1.1.2]
        ]
    """
    import json
    with open(filename) as jsonfile:
        lines = jsonfile.readlines()       
        counter = 0         # Variable to count the loop
        end = start+count   # End index
        for line in lines:
            if (counter >= start) and (counter < end):
                thread_json = json.loads(line)
                comments = flatten_thread2pairs_single(thread_json, comments)
            if (counter >= end):
                break
            if counter % 100 == 0:
                print('Processing thread %00000d' % counter)
            counter = counter + 1
    return comments


def flatten_thread2pairs_single(thread_json, comment_pairs=[]):
    """
    Flattens a single reddit thread into comment pairs
    1. the tree will be traversed in a depth first order
    2. if the comment has no body, it will be discarded
    
    Parameters
    ----------
    thread_json : dictionary
        A single reddit thread.
    comment_pairs : list of lists, optional
        See output. The default is [].

    Returns
    -------
    comment_pairs : list of lists
        [
            [comment1 comment1.1]
            [comment1.1 comment1.1.1]
            [comment1.1 comment1.1.2]
        ].

    """
    
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
                print("Can't find 'parent")
                pass
    return comment_pairs


def flatten_threads2single_all(filename, comments=[], start=0, count=100000):
    """
    Flatten entire reddit json file into a single level of comments
    1. the tree will be traversed in a depth first order
    2. if the comment has no body, it will be discarded

    Parameters
    ----------
    filename : string
        Name of a .json file.
    comments : list, optional
        List of comments. The default is [].
    start : int, optional
        Index to start from. The default is 0.
    count : int, optional
        Number of threads to do. The default is 100000.

    Returns
    -------
    comments : list
        [
            comment1
            comment1.1
            comment1.2
        ]

    """
    import json
    with open(filename) as jsonfile:
        lines = jsonfile.readlines()       
        counter = 0         # Variable to count the loop
        end = start+count   # End index
        for line in lines:
            if (counter >= start) and (counter < end):
                thread_json = json.loads(line)
                comments = flatten_thread2single_single(thread_json, comments)
            if (counter >= end):
                break
            if counter % 500 == 0:
                print('Processing thread %00000d' % counter)
            counter = counter + 1
    return comments


def flatten_thread2single_single(thread_json, comments=[]):
    """
    Flattens a single reddit reddit into a list of comments
    1. the tree will be traversed in a depth first order
    2. if the comment has no body, it will be discarded

    Parameters
    ----------
    thread_json : dictionary
        A reddit thread in dictionary form.
    comments : list, optional
        See output. The default is [].

    Returns
    -------
    comments : list
        [
            comment1
            comment1.1
            comment1.2
        ]

    """
    # posts_json is a list of dicts
    posts_json = thread_json['posts']
    
    for each_post in posts_json:    
        if not post_has_body(each_post):
            # Skip cauz the post has no body
            pass
        else:
            comments.append(each_post)
    return comments

"""
Some helper functions below
"""

def post_is_first(post_json):
    """
    Determines if post is the first in thread 
    Check whether the json field 'is_first_post' exists
    
    Parameters
    ----------
    post_json : dictionary
        A dictionary that is a reddit post.

    Returns
    -------
    bool
        True if post is_first_post.

    """
    if 'is_first_post' in post_json.keys():
        return True
    else:
        return False
    

def post_has_body(post_json):
    """
    Determine if post has text/body. If no, data is missing
    Check whether the json field 'body' exists
    
    Parameters
    ----------
    post_json : dictionary
        A dictionary that is a reddit post.

    Returns
    -------
    bool
        True if post has a body.

    """
    if 'body' in post_json.keys():
        return True
    else:
        return False    
    
    
def post_parent(single_post_json, posts_json, lut):
    """
    Finds the parent of current post.
    Just look at the 'in_reply_to' json field
    In some cases, the 'in_reply_to' is linked to a bot post
    In those cases, use 'majority_link' instead
    Other times, the comment is deleted from database. 
    
    Parameters
    ----------
    single_post_json : dictionary
        A post in dictionary format.
    posts_json : list
        List of dictionaries. All posts in the current thread.
    lut : list
        A list of post IDs.

    Returns
    -------
    parent : dictionary
        A post in dictionary format.

    """
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
        
        # Find that parent
        parent = posts_json[parent_index]
    # If trying majority link still doesnt give a parent
    # Just give up on this post. Throw error
    return parent

#TODO
def count_sentence(post):
    """
    Counts number of sentences in a post
    """
    return

#TODO
def count_tokens(post_json):
    """
    Counts the number of tokens in a post

    Parameters
    ----------
    post_json : dictionary
        A dictionary that is a reddit post.

    Returns
    -------
    count : int
        number of tokens in post.
    """
    tokenizer_text = tokenizer.tokenize(post_json['body'])
    count = len(tokenizer_text)
    return count

def count_words(sentence, tokenizer):
    """
    Counts number of words in a sentence

    Parameters
    ----------
    sentence : list
        DESCRIPTION.
    tokenizer : transformers.Tokenizer
        A tokenizer from huggingface's transformers library.

    Returns
    -------
    counts : int
        Number of words in sentence.
    
    """
    print("Sentence is: ", sentence)
    print('Tokenize input')
    
    tokenized_text = tokenizer.tokenize(sentence)
    return tokenized_text
    
def count_all_labels(comments):
    """
    Goes thru all the threads in dataset, then count the labels in a histogram

    Parameters
    ----------
    comments : list of dictionaries
        List of all posts in the thread in dictionary form.

    Returns
    -------
    histogram : dictionary
        key=labels, value=counts.
    errorloc : TYPE
        a list containing indices with no majority label.

    """
    histogram = {}
    counter = 0
    errorloc= []
    for each_comment in comments:
        #print(each_comment)
        try:
            label = each_comment['majority_type']
        except Exception:
            errorloc.append(counter)
            print('No majority type')
        # Does the label already exist in dictionary?
        if label in histogram.keys():
            # If yes, add 1 to the count
            histogram[label] = histogram[label] + 1
        else:
            # If no, start the count at 1
            histogram[label] = 1
        counter = counter + 1
    return histogram, errorloc

def count_all_token_lengths(comments):
    
    return 

def print_json_file(json_filename, start=0, count=5, debug=False):
    """
    Prints a few samples inside the json file
    Args:
        json_filename:  text of database file name
        start_index:    index to start printing from. default=0
        count:          how many items to print. default=5
        debug:          set to True if need to save the logfile
    Returns:
        None
    """
    import json
    if debug:
        logfile = open('logfile.txt', 'w')
    with open(json_filename) as jsonfile:
        counter = 0
        lines = jsonfile.readlines()
        counter = 0         # Variable to count the loop
        end = start+count   # End index
        for line in lines:
            if (counter >= start) and (counter < end):
                reader = json.loads(line) 
                # If this thread has posts, enter the thread
                # the dumps function helps to display json data nicely
                helper = json.dumps(reader, indent=4)
                print(helper)
                if debug:
                    logfile.write(helper)
            if counter >= end:
                break
            counter = counter + 1
    if debug:
        logfile.close()

if __name__ =='__main__':
    
    pairs = flatten_threads2pairs_all('coarse_discourse_dump_reddit.json')
    comments = flatten_threads2single_all('coarse_discourse_dump_reddit.json')
    
    histogram, errorlocations = count_all_labels(comments)
    histogram['unclear'] = len(errorlocations)
    
    labels = list(histogram.keys())
    counts = list(histogram.values())
    
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.bar(x=labels, height=counts)
    plt.ylabel('Counts')
    plt.xlabel('Labels')
    plt.xticks(labels, labels, rotation='vertical')
    plt.tight_layout()
    plt.grid(True)
    
    
    token_counts = []
    counter = 0
    for each_post in comments:
        if (counter % 1000 == 0):
            print("Tokenizing sample ", counter)
        token_count = count_tokens(each_post)
        token_counts.append(token_count)
        counter = counter + 1
    
    plt.figure(2)
    plt.hist(token_counts, bins=10,
             edgecolor='black', 
             facecolor='blue')
    plt.yscale('log')
    #plt.xscale('log')
    plt.ylabel('frequency')
    plt.xlabel('token length')
