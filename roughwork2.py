# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:11:50 2020

@author: Yong Keong
"""
import json 
'''
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

with open('coarse_discourse_dump_reddit.json') as jsonfile:
    counter = 0
    lines = jsonfile.readlines()
    
    for line in lines:
        if counter == 1:
            reader = json.loads(line) 
            prettyformat = json.dumps(reader, indent=4) 
            print(prettyformat)
            break
        counter = counter + 1
        