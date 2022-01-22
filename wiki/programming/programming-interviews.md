---
date: {}
title: Programming Interviews
published: true
---
## Google tips

This is a list of Google's tips to use when preparing for a job interview for a coding position.

### Algorithm Complexity
- Please review complex algorithms, including "Big O" notation.

### Sorting
- Know how to sort. Don't do bubble-sort.
- You should know the details of at least one n*log(n) sorting algorithm, preferably two (say, quicksort and merge sort). Merge sort can be highly useful in situations where quicksort is impractical, so take a look at it.

### Hash Tables
- Be prepared to explain how they work, and be able to implement one using only arrays in your favorite language, in about the space of one interview.

### Trees and Graphs
- Study up on trees: tree construction, traversal, and manipulation algorithms. You should be familiar with binary trees, n-ary trees, and trie-trees at the very least. You should be familiar with at least one flavor of balanced binary tree, whether it's a red/black tree, a splay tree or an AVL tree, and you should know how it's implemented.
- More generally, there are three basic ways to represent a graph in memory (objects and pointers, matrix, and adjacency list), and you should familiarize yourself with each representation and its pros and cons.
- Tree traversal algorithms: BFS and DFS, and know the difference between inorder, postorder and preorder traversal (for trees). You should know their computational complexity, their tradeoffs, and how to implement them in real code.
- If you get a chance, study up on fancier algorithms, such as Dijkstra and A* (for graphs).

### Other data structures:
- You should study up on as many other data structures and algorithms as possible. You should especially know about the most famous classes of NP-complete problems, such as traveling salesman and the knapsack problem, and be able to recognize them when an interviewer asks you them in disguise.

### Operating Systems, Systems Programming and Concurrency:
- Know about processes, threads, and concurrency issues. Know about locks, mutexes, semaphores, and monitors (and how they work). Know about deadlock and livelock and how to avoid them.
- Know what resources a processes needs, a thread needs, how context switching works, and how it's initiated by the operating system and underlying hardware.
- Know a little about scheduling. The world is rapidly moving towards multi-core, so know the fundamentals of "modern" concurrency constructs.

### Coding
- You should know at least one programming language really well, preferably C/C++, Java, Python, Go, or Javascript. (Or C# since it's similar to Java.)
- You will be expected to write code in your interviews and you will be expected to know a fair amount of detail about your favorite programming language.

### Recursion and Induction
- You should be able to solve a problem recursively, and know how to use and repurpose common recursive algorithms to solve new problems.
- Conversely, you should be able to take a given algorithm and prove inductively that it will do what you claim it will do.

### Data Structure Analysis and Discrete Math
- Some interviewers ask basic discrete math questions. This is more prevalent at Google than at other companies because we are surrounded by counting problems, probability problems, and other Discrete Math 101 situations.
- Spend some time before the interview on the essentials of combinatorics and probability. - You should be familiar with n-choose-k problems and their ilk – the more the better.

### System Design
- You should be able to take a big problem, decompose it into its basic subproblems, and talk about the pros and cons of different approaches to solving those subproblems as they relate to the original goal.

### Recommended Reading
- Google solves a lot of big problems; here are some explanations of how they solved a few to get your wheels turning.
  - Online Resources:
    - [Research at Google: Distributed Systems and Parallel Computing](http://research.google.com/pubs/DistributedSystemsandParallelComputing.html)
    - [Google File System](http://research.google.com/archive/gfs.html)
    - [Google Bigtable](http://research.google.com/archive/bigtable.html)
    - [Google MapReduce](http://research.google.com/archive/mapreduce.html)
- Algorithm Recommended Resources:
  - Online Resources:
    - [Topcoder - Data Science Tutorials](http://www.topcoder.com/tc?module=Static&d1=tutorials&d2=alg_index)
    - [The Stony Brook Algorithm Repository](http://www.cs.sunysb.edu/~algorith/)
  - Book Recommendations:
    - [Review of Basic Algorithms: Introduction to the Design and Analysis of Algorithms by Anany Levitin](https://www.google.com/webhp?hl=en&changed_loc=0#q=review+of+basic+algorithms+introduction+to+the+design+and+analysis+of+algorithms&hl=en&tbm=shop)
    - [Algorithms by S. Dasgupta, C.H. Papadimitriou, and U.V. Vazirani](http://www.cs.berkeley.edu/~vazirani/algorithms.html)
    - [Algorithms For Interviews by Adnan Aziz and Amit Prakash,](http://www.algorithmsforinterviews.com/)
    - [Algorithms Course Materials by Jeff Erickson](http://www.cs.uiuc.edu/~jeffe/teaching/algorithms)
    - [Introduction to Algorithms by Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest and Clifford Stein](http://mitpress.mit.edu/catalog/item/default.asp?ttype=2&tid=11866)
- Coding Book Recommendation:
  - [Programming Interviews Exposed; Secrets to landing your next job by John Monagan and Noah Suojanen](https://www.google.com/shopping/product/10904392385806396516?q=types+of+coding+questions+google+asks+programming+interviews+exposed&hl=en&biw=1745&bih=1005&site=webhp&sqi=2&pjf=1&bav=on.2,or.r_cp.&tch=1&ech=1&psi=539KVdiqEIvJsAWTkoBA.1430945767303.7&prds=paur:ClkAsKraXwLuomUnytrmklo3nqBglR3OsF49REA5hOKVeConNTghOhPlBuN07lUczldHXy82BXrpry53lNVyyMXa_ratGQnPKZRz5wGMWqi0YaxcUFWEj1j4WRIZAFPVH70DMoZJ2iytH9uRyKAQX_9d9ry0zw&ei=TYBKVcaOD8WzoQTbwIGQCQ&ved=0CF0QpiswAQ)


## Leetcode Tips

Leetcode must be a familiar platform to you if you are trying to find any software roles. However, there are just too many Leetcode questions and say, given only a month before your first interview, it is difficult to know where to start, even though you might get yourself conceptually familiar with all of the topics above. Therefore, in limtied time, the key to coding practice is **inductive learning**. Rather than spending a lot of time from the first question to question 200, you should do Leetcode questions topic by topic. The steps can be as follows:
- If you do not have a Leetcode premium subscription, find a reference which maintains a full list of categorized Leetcode questions. A good reference can be found [here](https://github.com/wisdompeak/LeetCode). Otherwise, with the premium subscription, you can sort the questions of interest by using the "tag" feature. 
- Choose a language: if you are mainly looking for roles in machine learning, or deep learning, stick with Python. Otherwise, for software positions in the robotics industry, C++ will be more popular. 
- List out topics you want to practice with priority assigned. For software positions in the robotics industry, the most important data stucture would be Graph, Hash Map, Stack, Queue, and Tree, and the most important algorithm you should have a grip of is DFS, BFS, Topological Sort, and some classic Dynamic Programming approaches (listed below). You should be familiar with the runtime complexity of BFS/DFS implemented in either iteration or recursion, as well as the pros and cons of implementation of recursion or iteration. Plus, the idea of DFS is not only applied to graphs, but problems that involve strings are also solved by DFS (e.g. [Permutation](https://leetcode.com/problems/permutations/) and [Combination](https://leetcode.com/problems/combinations/)). 
- Dynamic Programming (DP) is not very popular when compared to the other algorithms in technical interviews for robotics software engineers. Questions asked during interview when DP is the optimal solution are usually tailored for DP. For example, [Jump Game](https://leetcode.com/problems/jump-game/) (and its other variants too, e.g. [Jump Game II](https://leetcode.com/problems/jump-game-ii/)) and [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) are some classic (as well as popular) problems using 1-dimensional DP. [Unique Paths](https://leetcode.com/problems/unique-paths/) and [Dungeon Game](https://leetcode.com/problems/dungeon-game/solution/) are some classic problems using 2-dimensional DP that are encountered very frequently during interviews. 
- Create an excel sheet that records every question that you have completed, along with its related topic, runtime & space complexity, and, if possible, its difference compared to its other variants (e.g. Jump Game I to VII, Combination Sum I to IV). This is the cheat sheet you will cram for on the night before the interview. 


## Additional resources

* Daily plan for programming interview practice: https://github.com/jwasham/coding-interview-university#the-daily-plan
