Google Programming Tips
This is a list of Google's tips to use when preparing for a job interview for a coding position

Algorithm Complexity:

    Please review complex algorithms, including big O notation. For more information on algorithms, visit the links below and your friendly local algorithms textbook.
        Online Resources: __Topcoder - Data Science Tutorials__, __The Stony Brook Algorithm Repository__
        Book Recommendations: __Review of Basic Algorithms: Introduction to the Design and Analysis of Algorithms by Anany Levitin__, __Algorithms by S. Dasgupta, C.H. Papadimitriou, and U.V. Vazirani__, __Algorithms For Interviews by Adnan Aziz and Amit Prakash__, __Algorithms Course Materials by Jeff Erickson__, __Introduction to Algorithms by Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest and Clifford Stein__

Sorting:

    Know how to sort. Don't do bubble-sort.
    You should know the details of at least one n*log(n) sorting algorithm, preferably two (say, quicksort and merge sort). Merge sort can be highly useful in situations where quicksort is impractical, so take a look at it.

Hash Tables:

    Be prepared to explain how they work, and be able to implement one using only arrays in your favorite language, in about the space of one interview.

Trees and Graphs:

    Study up on trees: tree construction, traversal, and manipulation algorithms. You should be familiar with binary trees, n-ary trees, and trie-trees at the very least. You should be familiar with at least one flavor of balanced binary tree, whether it's a red/black tree, a splay tree or an AVL tree, and you should know how it's implemented.
    More generally, there are three basic ways to represent a graph in memory (objects and pointers, matrix, and adjacency list), and you should familiarize yourself with each representation and its pros and cons.
    Tree traversal algorithms: BFS and DFS, and know the difference between inorder, postorder and preorder traversal (for trees). You should know their computational complexity, their tradeoffs, and how to implement them in real code.
    If you get a chance, study up on fancier algorithms, such as Dijkstra and A* (for graphs).

Other data structures:

    You should study up on as many other data structures and algorithms as possible. You should especially know about the most famous classes of NP-complete problems, such as traveling salesman and the knapsack problem, and be able to recognize them when an interviewer asks you them in disguise.

Operating Systems, Systems Programming and Concurrency:

    Know about processes, threads, and concurrency issues. Know about locks, mutexes, semaphores and monitors, and how they work. Know about deadlock and livelock and how to avoid them.
    Know what resources a processes needs, a thread needs, how context switching works, and how it's initiated by the operating system and underlying hardware.
    Know a little about scheduling. The world is rapidly moving towards multi-core, so know the fundamentals of "modern" concurrency constructs.

Coding:

    You should know at least one programming language really well, preferably C/C++, Java, Python, Go, or Javascript. (Or C# since it's similar to Java.)
    You will be expected to write code in your interviews and you will be expected to know a fair amount of detail about your favorite programming language.
    Book Recommendation: __Programming Interviews Exposed; Secrets to landing your next job by John Monagan and Noah Suojanen (Wiley Computer Publishing)__

Recursion and Induction:

    You should be able to solve a problem recursively, and know how to use and repurpose common recursive algorithms to solve new problems.
    Conversely, you should be able to take a given algorithm and prove inductively that it will do what you claim it will do.

Data Structure Analysis and Discrete Math:

    Some interviewers ask basic discrete math questions. This is more prevalent at Google than at other companies because we are surrounded by counting problems, probability problems, and other Discrete Math 101 situations.
    Spend some time before the interview on the essentials of combinatorics and probability. You should be familiar with n-choose-k problems and their ilk – the more the better.

System Design:

    You should be able to take a big problem, decompose it into its basic subproblems, and talk about the pros and cons of different approaches to solving those subproblems as they relate to the original goal.
    Google solves a lot of big problems; here are some explanations of how we solved a few to get your wheels turning.
        Online Resources: __Research at__ __Google____: Distributed Systems and Parallel Computing__
        __Google__ __File System__
        __Google__ __Bigtable__
        __Google__ __MapReduce__
