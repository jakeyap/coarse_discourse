# TO DO LIST
To try to learn how to classify discourse into the following categories
0. question/request
1. answer
2. annoucement
3. agreement
4. positive reaction / appreciation
5. disagreement
6. negative reaction
7. elaboration / FYI
8. humor
9. other


## Task 0: Preprocessing
~~Import data from reddit~~
~~Flatten tree into single comments~~
~~Flatten the tree into pairs~~
~~Get a sense of category labels density~~
~~Remove deleted posts~~
~~Some first posts have no body. Deal with them by concatenating title with body~~

~~For 1st posts, combined title with body~~
~~Convert label pairs into numbers~~
~~Draw a histogram of token lengths for posts: 
- In progress

Tokenize the entire dataset

## Task 1: Try to learn relations from comment pairs first.
For example,
-post-1
    -comment-1.1
        -comment-1.1.1
        -comment-1.1.2
    -comment-1.2
        -comment-1.2.1
    -comment-1.3

will break down into
(post-1, comment-1.1)
(post-1, comment-1.2)
(post-1, comment-1.3)
(comment-1.1, comment-1.1.1)
(comment-1.1, comment-1.1.2)
(comment-1.2, comment-1.2.1)
Generalized as (commentA commentB)

After flattening, learn classification pairwise. 
There are 3 subtasks associated with Task 1.

Given (commentA, commentB),
### Task 1a: predict commentA and commentB labels simultaenously
### Task 1b: predict labelA, then labelB with labelA as a feature
### Task 1c: use labelA as a feature to predict labelB

Task 1a should be the hardest to train. 
Status: with a simple 3 layer NN after BERT, 10x10 category labels, I seem to be getting an accuracy of 50%. When I transform the features into 2D features, it is heavily skewed towards the question:answer label, so the NN gets lazy and ends up predicting Q:A for everything now.

Perhaps have to explore how to weigh the cost function, increase the batch size.

Task 1b is how humans really behave
Task 1c is the cheat way

## Task 2: 
Maintain tree structure. Not sure how yet.
