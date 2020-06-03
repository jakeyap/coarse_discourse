#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:16:05 2020
Some utility functions to help with data processing
@author: jakeyap
"""
import json
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import math

counter_first_post_no_bodyco = 0
counter_missing_parent = 0

def flatten_threads2pairs_all(filename, comments=[], start=0, count=100000, 
                              merge_title=True):
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
    errors : int
        Number of posts with no parent
    """
    import json
    with open(filename) as jsonfile:
        lines = jsonfile.readlines()       
        counter = 0         # Variable to count the loop
        end = start+count   # End index
        errors = 0          # Counter to store num of errors
        for line in lines:
            if (counter >= start) and (counter < end):
                thread_json = json.loads(line)
                try:
                    comments = flatten_thread2pairs_single(thread_json, 
                                                           comments,
                                                           merge_title)
                except Exception:
                    errors = errors + 1
            if (counter >= end):
                break
            if counter % 1000 == 0:
                print('Flattening thread to pair: %00000d' % counter)
            counter = counter + 1
    return comments, errors

def flatten_thread2pairs_single(thread_json, comment_pairs=[], 
                                merge_title=True):
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
    merge_title : Boolean. The default is True
        Decide merge title of thread into 1st post's body
    Raises
    ------
    Exception
        if no parent post is found.
        
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
            if merge_title:
                if post_has_body(each_post):
                    each_post['body'] = thread_json['title']+' '+each_post['body']
                else:
                    global counter_first_post_no_body
                    counter_first_post_no_body = counter_first_post_no_body + 1
                    each_post['body'] = thread_json['title']
        elif post_has_body(each_post):
            try:
                # Find current post's parent
                parent = post_parent(each_post, posts_json, lookuptable)
                # Store [parent, child] into the global list of lists
                comment_pairs.append([parent, each_post])
            except Exception:
                # sometimes, just cannot find parent post. skip
                global counter_missing_parent
                counter_missing_parent = counter_missing_parent + 1
        else:
            # Skip cauz the post has no body
            pass
            
    return comment_pairs

def flatten_threads2single_all(filename, comments=[], start=0, 
                               count=100000, merge_title=True):
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
    merge_title : Boolean. The default is True
        Decide merge title of thread into 1st post's body
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
                comments = flatten_thread2single_single(thread_json, comments,
                                                        merge_title)
            if (counter >= end):
                break
            if counter % 1000 == 0:
                print('Flattening thread to single: %00000d' % counter)
            counter = counter + 1
    return comments

def flatten_thread2single_single(thread_json, comments=[], merge_title=True):
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
    merge_title : Boolean. The default is True
        Decide merge title of thread into 1st post's body
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
        if post_is_first(each_post):
            if merge_title:
                if post_has_body(each_post):
                    each_post['body'] = thread_json['title']+' '+each_post['body']
                else:
                    each_post['body'] = thread_json['title']
        if post_has_body(each_post):
            comments.append(each_post)
        else:
            # Skip cauz the post has no body
            pass
    return comments

def filter_valid_pairs(comment_pairs):
    """
    Goes thru the comment_pairs, 
    rejects the ones that are have either one with no clear label
    rejects the ones with len(body) == 0 or deleted body
    
    Parameters
    ----------
    comment_pairs : list of list
        List that stores comment pairs.

    Returns
    -------
    filtered_comment_pairs : list of list
        Truncated version of comment_pairs.
        
    """
    
    filtered_comment_pairs = []
    for each_pair in comment_pairs:
        head = each_pair[0]
        tail = each_pair[1]
        # valid if head AND tail have proper label
        if (post_has_majority_label(head) and post_has_majority_label(tail)):
            if post_has_body(head) and post_has_body(tail):
                filtered_comment_pairs.append(each_pair)
    return filtered_comment_pairs

def filter_valid_comments(comments):
    """
    Goes thru comments,
    reject those (without label), (len(body)==0 or deleted), no parent

    Parameters
    ----------
    comments : list of dictionaries
        list of dictionaries, where each dictionary is a comment.

    Returns
    -------
    filtered_comments : list of dictionaries
        list of dictionaries, where each dictionary is a comment.

    """
    
    seen_ids = set()
    filtered_comments = []
    for each_comment in comments:
        if post_has_majority_label(each_comment) and post_has_body(each_comment):
            if post_is_first(each_comment):
                filtered_comments.append(each_comment)
                seen_ids.add(each_comment['id'])
            else:
                try:
                    parent_id = each_comment['majority_link']
                    if parent_id in seen_ids:
                        filtered_comments.append(each_comment)
                        seen_ids.add(each_comment['id'])
                except Exception:
                    print('No parent found. Skip post')
                    pass
                        
                '''
                try:
                    parent_id = each_comment['in_reply_to']
                    if parent_id in seen_ids:
                        filtered_comments.append(each_comment)
                        seen_ids.add(each_comment['id'])
                except Exception:
                    try:
                        parent_id = each_comment['majority_link']
                        if parent_id in seen_ids:
                            filtered_comments.append(each_comment)
                            seen_ids.add(each_comment['id'])
                    except Exception:
                        print('No parent found. Skip post')
                        pass
                        '''
    return filtered_comments
        
def filter_first_comments(comments):
    """
    Goes thru comments, spits out a list of comments which are 1st posts

    Parameters
    ----------
    comments : list of dictionaries
        list of dictionaries where each comment is a dictionary

    Returns
    -------
    filtered_comments : list of dictionaries
        list of dictionaries where each comment is a dictionary

    """
    filtered_comments = []
    for each_comment in comments:
        if post_is_first(each_comment):
            filtered_comments.append(each_comment)
    return filtered_comments
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
    Check whether body has non zero length
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
        body = post_json['body']
        if (body == "" or body=="[deleted]"):
            return False
        else:
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

def post_has_majority_label(post_json):
    """
    Determines if post has a majority label
    Check whether the json field 'majority_type' exists

    Parameters
    ----------
    post_json : dictionary
        A dictionary that is a reddit post.

    Returns
    -------
    bool
        True if post has a majority_type.

    """
    if 'majority_type' in post_json.keys():
        return True
    else:
        return False  
    
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
    print('Loading pre-trained model tokenizer')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer_text = tokenizer.tokenize(post_json['body'])
    count = len(tokenizer_text)
    return count

def count_all_labels(comments):
    """
    Goes thru all the threads in dataset, 
    then count the labels in a histogram

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
    histogram = empty_label_dictionary()
    counter = 0
    errorloc= []
    for each_comment in comments:
        #print(each_comment)
        try:
            label = each_comment['majority_type']
        except Exception:
            errorloc.append(counter)
            #print('No majority type')
        # Does the label already exist in dictionary?
        if label in histogram.keys():
            # If yes, add 1 to the count
            histogram[label] = histogram[label] + 1
        else:
            # If no, start the count at 1
            histogram[label] = 1
        counter = counter + 1
    return histogram, errorloc

def count_all_labels_pairs(valid_comment_pairs):
    """
    Goes thru all the comment pairs in dataset, 
    then count the labels in a histogram

    Parameters
    ----------
    valid_comment_pairs : list of list
        Truncated version of comment_pairs.

    Returns
    -------
    histogram : dictionary
        key=labels, value=counts.
    seen_ids : set
        set of unique IDs of posts that were explored.
        
    """
    
    # Set to store seen posts
    seen_ids = set()
    # Dictionary to store histogram
    histogram = empty_label_dictionary()
    for each_pair in valid_comment_pairs:
        head = each_pair[0]
        tail = each_pair[1]
        
        head_id = head['id']
        tail_id = tail['id']
        
        # If head is not seen before
        if not (head_id in seen_ids):
            # Add to set
            seen_ids.add(head_id)   
            
            # Find the data's label
            label = head['majority_type']
            
            # Add to count or start a count
            if label in histogram.keys():
                # If yes, add 1 to the count
                histogram[label] = histogram[label] + 1
            else:
                # If no, start the count at 1
                histogram[label] = 1
            
        if not (tail_id in seen_ids):
            # Add to set
            seen_ids.add(tail_id)   
            
            # Find the data's label
            label = tail['majority_type']
            
            # Add to count or start a count
            if label in histogram.keys():
                # If yes, add 1 to the count
                histogram[label] = histogram[label] + 1
            else:
                # If no, start the count at 1
                histogram[label] = 1
            
    return histogram, seen_ids
    
def count_all_token_lengths(comments):
    """
    Goes thru all comments, counts the number of tokens inside

    Parameters
    ----------
    comments : list of dictionaries
        List of all posts in the thread in dictionary form.

    Returns
    -------
    token_counts : list of int
        a list where each element corresponds to the number of tokens in 
        a post at the same index in the comments list.

    """
    token_counts = []
    counter = 0
    for each_post in comments:
        if (counter % 1000 == 0):
            print("Tokenizing & counting sample ", counter)
        token_count = count_tokens(each_post)
        token_counts.append(token_count)
        counter = counter + 1
    return token_counts

def print_json_file(json_filename, start=0, count=5, debug=False):
    """
    Pretty prints a few samples inside the json file

    Parameters
    ----------
    json_filename : string
        text of database file name.
    start : int, optional
        Index to start printing from. The default is 0.
    count : TYPE, optional
        How many items to print. The default is 5.
    debug : Boolean, optional
        True if need to save the logfile. The default is False.

    Returns
    -------
    None.

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

def print_json_comment(comment):
    """
    Pretty prints a reddit comment

    Parameters
    ----------
    comment : string

    Returns
    -------
    None.

    """
    print(json.dumps(comment, indent=4))

def print_json_comment_pair(comment_pair):
    """
    Pretty prints a pair of reddit comments

    Parameters
    ----------
    comment_pair : list of lists
        [commentA, commentB].

    Returns
    -------
    None.

    """
    print(json.dumps(comment_pair, indent=4))

def empty_label_dictionary():
    """
    Creates a dictionary of labels:counts

    Returns
    -------
    categories : dictionary
        Dictionary containing the counts of the labels.

    """
    categories = {'question':0,
                  'answer':0,
                  'announcement':0,
                  'agreement':0,
                  'appreciation':0,
                  'disagreement':0,
                  'negativereaction':0,
                  'elaboration':0,
                  'humor':0,
                  'other':0}
    return categories

def find_post_wo_body(json_filename, start=0, stop=1000000):
    """
    Finds the index of a post without a body field or an empty body content
    Returns the first index that matches the search criteria
    
    Parameters
    ----------
    json_filename : string
        text of database file name..
    start : int, optional
        index to start counting from. The default is 0.
    stop : int, optional
        index to stop at. The default is 1000000.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    counter = start
    with open(json_filename) as jsonfile:
        threads_string = jsonfile.readlines()
        for thread_string in threads_string[start:stop]:
            thread_json = json.loads(thread_string)
            firstpost = thread_json['posts'][0]
            try:
                if (firstpost['body'] == ''):
                    return counter
            except Exception:
                return counter
            counter = counter + 1
    return -1

def count_firstposts_wo_body(comments):
    """
    Counts the number of posts that have no body content or field

    Parameters
    ----------
    comments : list of dictionaries
        List of all posts in the thread in dictionary form.

    Returns
    -------
    counter : int
        number of first posts without a body field or body content.

    """
    counter = 0
    for each_comment in comments:
        if post_has_body(each_comment):
            pass
        else:
            counter = counter + 1
    return counter

def filter_leaf_nodes(comments):
    """
    counts how deep a branch goes. do it by looking at the "depth field" 
    how it works: 
        1st, record all reply link IDs
        2nd, for any post, if no link to it exists, it is a leaf node
    
    do step 1 by storing 'majority_link' & 'in_reply_to' fields in a set
    do step 2 by searching thru the set
    
    Parameters
    ----------
    comments : list of dictionaries
        list of dictionaries where each dict is a comment

    Returns
    -------
    filtered_comments : list of dictionaries
        list of dictionaries where each dict is a leaf comment

    """
    
    seen_ids = set()
    filtered_comments = []
    # do step 1 first. create the lookup table of reply links
    for each_comment in comments:
        if 'majority_link' in each_comment.keys():
            seen_ids.add(each_comment['majority_link'])
        if 'in_reply_to' in each_comment.keys():
            seen_ids.add(each_comment['in_reply_to'])
    
    # do step 2. go thru all comments, see whether it is inside the list
    for each_comment in comments:
        if each_comment['id'] in seen_ids:
            pass
        else:
            filtered_comments.append(each_comment)
    return filtered_comments

def tree_depth_histogram(leaf_comments):
    """
    Creates a histogram of leaf nodes' depth

    Parameters
    ----------
    leaf_comments : list of dictionaries
        list of dictionaries where each dict is a leaf comment.

    Returns
    -------
    histogram : tuple of 2 arrays
        (counts, bin_edges).

    """
    
    counts = []
    for each_comment in leaf_comments:
        try:
            counts.append(each_comment['post_depth']+1)
        except Exception:
            counts.append(1)
    num_bins = max(counts)
    histogram = np.histogram(counts, bins=num_bins, range=(0,num_bins))
    return histogram

def comments_to_pandas(filtered_comments):
    """
    Convert a list of comments into pandas. 

    Parameters
    ----------
    filtered_comments : list of dictionaries
        list of dictionaries where each dict is a comment

    Returns
    -------
    Pandas dataframe

    """
    return pd.DataFrame(filtered_comments)

def pandas_remove_nan(dataframe):
    """
    Goes thru the rows in pandas dataframe, remove NaNs in data depth & 
    is_first_post fields

    Parameters
    ----------
    dataframe : pandas dataframe

    Returns
    -------
    dataframe : pandas dataframe

    """
    datalength = len(dataframe)
    for i in range(datalength):
        if math.isnan(dataframe.at[i, 'post_depth']):
            dataframe.at[i, 'post_depth'] = 1
        else:
            dataframe.at[i, 'post_depth'] = dataframe.at[i, 'post_depth'] + 1
            dataframe.at[i, 'is_first_post'] = False
    return dataframe

def pandas_find_parent_index(parent_id, dataframe):
    ''' given a parent_id, return the index inside the dataframe '''
    index = -1
    for i in range(len(dataframe)):
        test_id = dataframe.at[i, 'id']
        if (test_id == parent_id):
            index = i
            break
    return index

def pandas_find_post_label_str(index, dataframe):
    return dataframe.at[index, 'majority_type']

def pandas_find_post_label_num(index, dataframe):
    return dataframe.at[index, 'number_labels']

"""

if __name__ =='__main__':
    DATADIR = './data/'
    FILENAME = 'coarse_discourse_dump_reddit.json'
    #pairs, errors = flatten_threads2pairs_all(DATADIR+FILENAME)
    comments = flatten_threads2single_all(DATADIR+FILENAME)
    
    # remove posts with no body, no links, no labels
    valid_comments = filter_valid_comments(comments)
    # convert into pandas
    panda_comments = comments_to_pandas(valid_comments)    
    # tokenize each sentence and append to pandas dataframe
    '''
    histogram1, errorlocations = count_all_labels(comments)
    
    labels1 = list(histogram1.keys())
    counts1 = list(histogram1.values())
    
    #valid_comment_pairs = filter_valid_pairs(pairs)
    valid_comments = filter_valid_comments(comments)
    first_comments = filter_first_comments(valid_comments)
    #histogram2, _ = count_all_labels_pairs(valid_comment_pairs)
    histogram2, _ = count_all_labels(first_comments)
    labels2 = list(histogram2.keys())
    counts2 = list(histogram2.values())
    
    width=0.3
    import matplotlib.pyplot as plt
    plt.figure(1)
    xpts = np.arange(len(histogram1))
    #plt.bar(x=xpts+width/2, height=counts1, width=width, label='unfiltered')
    plt.bar(x=xpts-width/2, height=counts2, width=width, label='filtered')
    plt.ylabel('Counts')
    plt.xlabel('Filtering removes [deleted] or empty posts')
    plt.xticks(ticks=xpts, labels=labels1, rotation='vertical')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.grid(True)
    '''

"""