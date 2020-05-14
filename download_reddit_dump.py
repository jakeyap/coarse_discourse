# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:24:21 2020

@author: Yong Keong

1. Create a reddit account first
2. Go to https://www.reddit.com/prefs/apps, then create a new script app
3. take note of the client ID, client secret, userID, password

"""
import sys
sys.path.insert(0, './join_forum_data')

import praw

import reddit_tokens as passwords
userID = passwords.userID
userPW = passwords.userPW
clientID = passwords.clientID
clientSecret = passwords.clientSecret

# praw wrapper helps to grab token, sign in with token
reddit = praw.Reddit(client_id=clientID,
                     client_secret=clientSecret,
                     password=userPW,
                     user_agent="testscript",
                     username=userID)

# this should print username
print(reddit.user.me())


