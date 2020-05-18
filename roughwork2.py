# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:11:50 2020

@author: Yong Keong
"""
import json 
import reddit_utilities as rd_util
import time
'''
# import json file into python data
with open('coarse_discourse_dataset.json') as jsonfile:
    counter = 0
    lines = jsonfile.readlines()
    
    for line in lines:
        reader = json.loads(line) 
        #reader = json.dumps(line) 
        print(reader['url'])
        counter = counter + 1
        if counter > 10:
            break

# the dumps function helps to display json data nicely
helper = json.dumps(reader, indent=4)
print(helper)
'''        
DEBUG = False
STOP_INDEX = 100000
DEBUG_LINE = 845

# start flattening process
comment_pairs = []
time0 = time.time()
with open('coarse_discourse_dump_reddit.json') as jsonfile:
    counter = 0
    lines = jsonfile.readlines()
    
    for line in lines:
        if DEBUG:
            if counter == DEBUG_LINE:
                thread_json = json.loads(line)
                prettyformat = json.dumps(thread_json, indent=4) 
                print(prettyformat)
                debuglog = open('debuglog.txt', 'w')
                debuglog.write(prettyformat)
                debuglog.close()
                comment_pairs = rd_util.flatten_posts_pairs(thread_json, comment_pairs)
                break
        else:
            if counter < STOP_INDEX:
                thread_json = json.loads(line)
                comment_pairs = rd_util.flatten_posts_pairs(thread_json, comment_pairs)
            if counter >= STOP_INDEX:
                break
        if counter % 100 == 0:
            print('Processing thread %00000d' % counter)
        counter = counter + 1

time1 = time.time()
print('time taken is %0.2f s' % (time1-time0))
# comment_pairs  = rd_util.flatten_posts(line_json, comment_pairs)
