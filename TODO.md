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


## 1st task: Try to learn relations from comment pairs first.
Flatten the tree into pairs
Throw away tree structure and discard longer distance relationships 1st

As an example,
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

Try to learn classification based on this.
