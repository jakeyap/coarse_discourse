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
~~Flatten the tree into pairs~~
Convert labels into one-hot vectors
Draw a histogram of token lengths for posts
Decide on a length to truncate for BERT
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
### Task 1a: don't use commentA's label, predict labelB
### Task 1b: predict labelA, then labelB with labelA as a feature
### Task 1c: use labelA as a feature to predict labelB

Task 1a should be the hardest to train. 
Task 1b is how humans really behave
Task 1c is the cheat way

## Task 2: 
Maintain tree structure. Not sure how yet.
