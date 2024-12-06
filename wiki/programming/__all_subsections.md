/wiki/programming/boost-library/
---
date: 2017-08-21
title: Boost C++ Libraries
---
[![Boost Logo](assets/BoostLibrary-f962f.png)](https://www.boost.org/)

Boost is the most popular and widely used C++ library. It is available online for free. This stable C++ library contains many helpful data structures, algorithms, utilities, and more. This library influenced the design and implementation of the newest C++ library standard (C++11).

## Advantages
Key items to look at:
- [Shared Pointers](http://www.boost.org/doc/libs/1_55_0/libs/smart_ptr/shared_ptr.htm)
  - Shared pointers (boost::shared_ptr) provide a simple way to minimize memory leaks in C++ code. Replacing the old fashioned "Foo bar = new Foo" syntax with a boost::shared_ptr no longer requires you to delete the allocated memory, and the shared_ptr overhead is extremely small. An example usage is shown in the shared_ptr link above.
- [Mutex](http://www.boost.org/doc/libs/1_55_0/doc/html/thread/synchronization.html)
  - Boost offers a variety of Mutex data structures that are safe, effective, and easy to use. There are several types, which serve different purposes. Read up on boost mutex if you plan on using threads and are at all worried about thread safety.
- [Thread](http://www.boost.org/doc/libs/1_55_0/doc/html/thread.html)
  - Boost has an easy to use thread library (compared to standard pthread). This library allows for all of the standard threading functionality that pthread allows, but in an easier to use interface, along with some added functionality.


/wiki/programming/boost-maps-vectors/
---
date: 2017-08-21
title: Iterations in maps and vectors using boost libraries
---
Using Boost libraries for maps and vectors in C++ for ROS.

Maps are hash tables used to store data along with ids. For example if you are working on multiple robots and implement a navigation algorithm in which the controls of robot motion are dependent on robot pose. So one can create a map which maps robot ids to robot pose. This is how you do it:

``std::map<int,Eigen::Affine3d> Poses``

Now suppose you receive the pose of each of the robot, along with their ids from sensor data processing function or (an Apriltags function). Instead of iterating through the map using a for loop or stl iterators once, you can simply use boost library for `std::map`. Below is an example implementation.

```
#include <boost/foreach.hpp>
#include <boost/algorithm/clamp.hpp>
#include <boost/range/adaptor/map.hpp>

BOOST_FOREACH(const int i, robot_ids | boost::adaptors::map_values)
{
first check if the robot id is valid
if( Poses.find(i)!=Poses.end())
{
update poses
Poses(i)= pose_received_from_function;
}
}
```
[Here is the link](http://cplusplus.bordoon.com/boost_foreach_techniques.html) to tutorial on how to use maps and boost libraries.  Boost libraries are recommended for those who  would use maps.


/wiki/programming/cmake/
---
date: 2017-08-21
title: CMake and Other Build Systems
---
Programs written in compiled languages like C/C++ must be compiled before being run, a process which converts the high-level code written by the programmer into binary files that can be executed on a particular architecture. A build system is a framework that automates the process of compiling files in a project with a specified compiler, and also links predefined libraries of code that are referenced in the project files. By using a build system, a developer can reduce the process of compiling a project to a single line in a terminal, avoiding the redundant steps of managing each file independently.

Popular build tools include Autotools, Make, Maven, Gradle, and Ninja. Each of these frameworks contains a set of protocols for managing project compilation, and the developer can specify the exact requirements, targets, and dependencies for their project. The Make build system refers to these specifications as a Makefile, although other build systems use other terms.

Writing a Makefile can be a complicated task, especially for larger projects. CMake is a tool that simplifies the generation of Makefiles for Make, Android Studio, Ninja, and others. CMake is open-source, and is available for Windows, Mac, and Linux. When a project that uses CMake is built, CMake will first generate a Makefile, and then that file will be used to instruct the build system in executing the project compilation.

When using CMake, it is common and accepted practice to organize a project folder according to a specific folder structure, as seen in the image below.

![CMake File Structure](assets/cmake_file_structure.png)

- **bin:** contains the executable files which can be run on the computer.
- **build:** contains the makefiles which are required to build the project.
- **CMakeLists.txt:** the script that CMake uses to generate the makefiles. Also references additional CMakeLists files in **src** to generate additional makefiles for subsections of the project.
- **data:** contains any data required by the project.
- **include:** stores a header file for each of the project files in **src**.
- **lib:** contains the static libraries which are linked to the appropriate executable files.
- **LICENSE:** a text or binary file that verifies the project's authenticity.
- **README.md:** the text file that describes the project and any additional information needed to run or build it.
- **src:** contains the project files that need to be compiled into executables and also the project files that act as libraries.
- **tools:** contains any scripts that need to be executed only once during the project life cycle. This may involve scripts to download and/or process data.

One of the features that CMake provides is that a project can have multiple CMakeLists.txt files. Each of these is a script for defining the rules to compile one or more files in the project. When the project is built, each of these scripts can be invoked from the outermost CMakeLists.txt file. A project can have multiple layers of Cmake scripts in this manner. This hierarchy of scripts allows a developer to easily manage large projects that depend on external libraries and packages. It is considered good practice to keep individual scripts short, and use this hierarchical structure to organize different modules of the project.

The nature of CMake scripts makes it simple to incorporate external libraries into a project. If the external library has a CMakeLists.txt file of its own, this file can be referenced in the outermost CMake script of the project. This will also compile any changes made to the source files of the external package without having to build that package separately. We can include the headers from the external library the same way we include our custom header files.


The outer CMakeLists file configures the project at a high level, including the version of CMake being used, the name of the project, and any global properties that are used to organize the build output. This file will also contain a list of all subdirectories that contain inner CMake scripts. The bulk of the work is done by these inner files, which specify which target source files will be converted into executables and which will be used as libraries, and link target files with the appropriate libraries.

When it is time to initialize the build process, the outer level CMakeLists.txt is generally invoked from inside the build directory by using  ‘cmake ..’. Next, the command ‘make’ should be sent in order to actually build the project.

Some of the most common and useful functions employed in CMake scripts are as follows:

Within the inner CMake scripts, the path to the outer level CMake script can be referenced using
```
${CMAKE_SOURCE_DIR}
```

The inner level scripts can be added to the top-level script by using the command
```
add_subdirectory(path_to_script)
```

The directory containing the project header files can be specified by using
```
include_directories(directory1, directory2, ...., directoryN)
```

The program files to be used as libraries can be specified using
```
add_library(library_name STATIC
  path_to_file1
  path_to_file2
  ....
  path_to_fileN)
```

The program files to be converted into executables can be specified using
```
add_executable(exec_name path_to_src_file)
```

Libraries can be linked to executables by using
```
target_link_libraries(exec_name PUBLIC library_name)
```

It is recommended to develop a familiarity with and understanding of the various functions provided by CMake, rather than merely copying code snippets found online. For a full description of the components of a CMakeLists.txt file, reference [this tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html) by the official CMake website.

CMake also provides a GUI that can be used to assemble scripts without writing them by hand. This GUI provides interfaces to designate modules as executables and libraries, and to link libraries to relevant files. The project can then be built from within the GUI, bypassing the need to interact with the terminal. This GUI option is popular for developers who work on Windows, where the command prompt is used less frequently.

# Recommended Reading
- Offical CMake tutorial: https://cmake.org/cmake/help/latest/guide/tutorial/index.html
- John Lamp's tutorials: https://www.johnlamp.net/cmake-tutorial.html
- A more basic but hand-on tutorial: http://mirkokiefer.com/blog/2013/03/cmake-by-example/
- Derek Molloy's BeagleBone tutorial: http://derekmolloy.ie/hello-world-introductions-to-cmake/


/wiki/programming/eigen-library/
---
date: 2017-08-21
title: How to use Eigen Geometry library for c++
---

The Eigen Geometry library is a C++ libary useful for robotics. It is capable of the following operations:
1. Declare Vectors, matrices, quaternions.
- Perform operations like dot product, cross product, vector/matrix addition ,subtraction, multiplication.
- Convert from one form to another. For instance one can convert quaternion to affine pose matrix and vice versa.
- Use AngleAxis function to create rotation matrix in a single line.

## Example Implementation
To use the library, the following includes are recommended:
```
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Core>
```
For instance, a rotation matrix homogeneous transform of PI/2 about z-axis can be written
as:

``Eigen::Affine3d T_rt(Eigen::AngleAxisd(M_PI/2.0, Eigen::Vector3d::UnitZ()));``

Additionally, you can:
1. Extract rotation matrix from Affine matrix using `Eigen::Affine3d Mat.rotation( )`
- Extract translation vector from Affine Matrix using `Eigen::Affine3d Mat.translation( )`
- Find inverse and transpose of a matrix using `Mat.inverse( ) and Mat.transpose( )`

The applications are the following
1. Convert Pose to Quaternions and vice versa
2. Find the relative pose transformations by just using simple 3D homogeneous transformation `Eigen::Affine3d T` is a 4*4 homogeneous transform:
![Homogeneous Equation Example](assets/EigenLibrary-35715.png)
3. Now all the transformations (rotation or translation) can be represented in homogeneous form as simple 4*4 matrix multiplications.
4. Suppose you have a pose transform T of robot in the world and you want to find robot’s X-direction relative to the world. You can do this by using
`Eigen::Vector3d x_bearing= T.rotation * Eigen::Vector3d::UnitX();`

## References
This is an important library in c++ which gives capabilities equal to Python for vectors and matrices. More helpful functions and examples can be found at the following links
- Eigen Documentation: http://eigen.tuxfamily.org/dox/
- Eigen Quaternion Documentation: https://eigen.tuxfamily.org/dox/classEigen_1_1Quaternion.html
- Eigen Transforms Documentation: https://eigen.tuxfamily.org/dox/classEigen_1_1Transform.html


/wiki/programming/git/
---
date: 2017-08-21
title: Git
---
Git is a distributed revision control and source code management system with an emphasis on speed. Every Git working directory is a full-fledged repository with complete history and full version tracking capabilities, and is not dependent on network access or a central server. Git is free software distributed under the terms of the GNU GPLv2.

Git is primarily a command-line tool but there are a few really good desktop applications that make it easier to work with (at the cost of hiding the advanced features). Working with a GUI can be hugely beneficial to mitigate the steep learning curve, but the recommended way to use git is using the command-line interface or CLI.

## Free Repository Providers
- ### [GitHub](https://www.github.com/)
  A well-supported and popular version control provider. Their desktop application [GitHub Desktop](https://desktop.github.com/) is high-quality, feature-rich GUI for Windows and Mac. GitHub offers unlimited public and private repositories and ability to create an 'organization' for free to host team projects, with limited monthly credits for automated builds. GitHub excels at providing a great experience around git and source code management, with good options for builds and deployment integrations.

  [GitHub Education Pack](https://education.github.com/pack) provides access to a bundle of popular development tools and services which GitHub and its partners are providing for free (or for a discounted price) to students.
- ### [GitLab](https://gitlab.com/explore)
  The other big player in the version control space and the first choice for Enterprise git deployments. Offers free unlimited public and private repositories with unlimited collaborators. Offers some nifty features that GitHub doesn't offer in their free tier such as protected branches, pages and wikis. Offers a self-hosting option if you want to run your own git server. Their platform is geared towards serving a complete and comprehensive DevOps workflow that is not just restricted to source code management.

- ### [BitBucket](https://bitbucket.org/)
  Another popular service, unlimited private repositories for up to 5 collaborators.
  - [Getting Started Guide](https://confluence.atlassian.com/display/BITBUCKET/Bitbucket+101)

## Popular GUI Clients
- [GitHub Desktop](https://desktop.github.com/)
  High-quality, well-supported GUI for Windows & Mac.
- [SourceTree](https://www.sourcetreeapp.com/)
  A feature-rich GUI for Windows & Mac.
- [GitKraken](https://www.gitkraken.com/)
  A powerful git client for Linux, Windows & Mac.
- [GitExtensions](https://gitextensions.github.io/)
  Also supports Linux, Windows & Mac.

## Learning Resources

### Basics
- [GitHub Learning Lab](https://lab.github.com/) offers some excellent courses on mastering the basics of git on their platform. Their [Introduction to GitHub](https://lab.github.com/githubtraining/introduction-to-github) course is great place to get started.
- [GitHub's Getting Started Guide](https://help.github.com/)
  Walks you through creating a repository on GitHub and basics of git.

- [Learn Git Branching](https://learngitbranching.js.org/):
  A browser-based game designed to introduce you to some of the more advanced concepts.

- [Git Immersion](http://gitimmersion.com/):
A hands-on tutorial that sets you up with a toy project and holds your hand through project development. Contains many useful aliases and shortcuts for faster workflow.

- [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials):
One of the best reading resources around git and version control. Contains a very good set of tutorials but more importantly has a comprehensive set of articles around the important topics in git and even the history of version control systems. Focuses on providing a detailed explanation of how git works rather than simply listing the commands.

### Intermediate

- [Managing Merge Conflicts](https://lab.github.com/githubtraining/managing-merge-conflicts)
- [Reviewing Pull Requests](https://lab.github.com/githubtraining/reviewing-pull-requests)
- [GitHub Actions Basics](https://lab.github.com/githubtraining/github-actions:-hello-world)
- [Cherry Picking Commits](https://www.atlassian.com/git/tutorials/cherry-pick)
- [Rebasing Branches](https://docs.github.com/en/get-started/using-git/about-git-rebase)
- [Git Blame](https://www.atlassian.com/git/tutorials/inspecting-a-repository/git-blame)

### Advanced
- [Git Reflog](https://www.atlassian.com/git/tutorials/rewriting-history/git-reflog)
- [Git Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Hooks, Enforcing Commit-Message Formats & ACLs](https://git-scm.com/book/en/v2/Customizing-Git-An-Example-Git-Enforced-Policy)
- [A Beginner's Guide to Git Bisect](https://www.metaltoad.com/blog/beginners-guide-git-bisect-process-elimination)
- [Continuous Integration Using GitHub Actions](https://lab.github.com/githubtraining/github-actions:-continuous-integration)

## References & Help
- [Atlassian Git Cheat Sheet](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet): A handy set of commands to have around your desk for those quick command look ups.
- [How to Undo Almost Anything with Git](https://github.blog/2015-06-08-how-to-undo-almost-anything-with-git/): A blog post from GitHub that lists some scary scenarios in git and how to undo almost anything.
- [Dangit, Git!?](https://dangitgit.com/): Quick references for getting out of bad situations in git. Very powerful set of fixes, but doesn't provide a good explanation of what happened / how the fix works - read before you blindly follow the instructions.
- [Official Git Documentation](http://git-scm.com/documentation)
Extremely thorough documentation of git. It can be quite dense if you're new to git, but this the most authoritative and updated documentation for git CLI usage. Best used to look up command-line usage, parameters and their rationale.
- Man pages: The git command line interface ships with a very good set of man pages accessible by running `man git` on the terminal and is available wherever git is installed. If you're not familiar with man pages, you can read about it [here](https://itsfoss.com/linux-man-page-guide/).


/wiki/programming/multithreaded-programming/
---
date: 2017-08-21
title: Multithreaded Programming as an Alternative to ROS
---
As a software framework for robotics, ROS is an an obvious choice. Having said that, sometimes ROS can be an overkill or can cause trouble due to the amount of abstraction it has. The option of using ROS should be carefully evaluated, especially when you have all your processing on one single embedded system/processor/computer and not distributed across multiple systems. For such a system, a logical alternative is implementing a C/C++ program from scratch and using libraries like pthreads, boost, or others for parallel/pseudo-parallel execution and other functionalities.

C/C++ on \*nix systems has POSIX threads or pthreads as they are popularly known. Pthreads are very powerful and allow interfaces to make any execution multithreaded (of course, the number of threads will be limited by the embedded system being used). Pthreads allow you to call a function/method asynchronously and then execute on its own dedicated thread. Moreover, you can also use them to parallelize your execution.

Here's an example of pthreads from Wikipedia:
```
[[code format="de1"]]
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define NUM_THREADS 5

void *perform_work(void *argument)
{
int passed_in_value;

passed_in_value = *((int *) argument);
printf("Hello World! It's me, thread with argument %d!\n", passed_in_value);

/* optionally: insert more useful stuff here */

return NULL;
}

int main(void)
{
pthread_t threads[NUM_THREADS];
int thread_args[NUM_THREADS];
int result_code, index;

create all threads one by one
for (index = 0; index < NUM_THREADS; ++index) {
thread_args[index] = index;
printf("In main: creating thread %d\n", index);
result_code = pthread_create(&threads[index], NULL, perform_work, (void *) &thread_args[index]);
assert(0 == result_code);
}

wait for each thread to complete
for (index = 0; index < NUM_THREADS; ++index) {
// block until thread 'index' completes
result_code = pthread_join(threads[index], NULL);
printf("In main: thread %d has completed\n", index);
assert(0 == result_code);
}

printf("In main: All threads completed successfully\n");
exit(EXIT_SUCCESS);
}
```



## Example case
In this example case, we would read, process, and store IMU values at a constant rate.

Say your system interfaces to an IMU which gives values at 200 Hz. One of the threads can read these values and process and store it in some shared memory along with timestamps of your values. More specifically, you call a function which has a while loop that looks for data on serial interface and then operates on that data (say integrate to get YPR from gyro values) and then store these YPR values in a buffer. Your main thread can use these YPR values whenever required by reading from this buffer.

If ROS is used you will have a separate node that interfaces to the IMU and then publishes the value as a tf/pose at a configurable rate. However, you will in most cases have to translate the message into a form that your other nodes can use. Moreover, you will have the additional queue and callback overhead of ROS which will be practically out of your control.


## Further Reading
1. Pthreads tutorial: https://computing.llnl.gov/tutorials/pthreads/
  - This is an exhaustive tutorial describing pthreads and all the functionalities to provides.
2. Pthreads tutorial for multithreading in C++ and Linux: http://www.tutorialspoint.com/cplusplus/cpp_multithreading.htm
  - This is a more application-oriented tutorial with examples.


/wiki/programming/programming-interviews/
---
date: 2022-1-26
title: Programming Interviews
published: true
---

This article is meant for people looking for software positions in the robotics industry. It introduces the data structure, algorithms, and other related topics you need to know in order to prepare for your technical interview. It also provides a detailed instruction on how to crack your coding sessions.

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
- For each topic in the list: sort the questios by the tag first, then sort by its frequency. Complete top 10~50 frequent questions and move on to the next topic in your list. (Note: premium subscription is required to see a question's frequency, but you can easily "bypass" this by asking for a screenshot from any of your friends who has premium subscription) 
- Create an excel sheet that records every question that you have completed, along with its related topic, runtime & space complexity, and, if possible, its difference compared to its other variants (e.g. Jump Game I to VII, Combination Sum I to IV). This is the cheat sheet you will cram for on the night before the interview. 
- Do mock interviews with peers. This will be very helpful if you are the type of person whose mindset will be influenced by the stress, or tension during an interview. Also mock interviews will improve your communication abilities during coding, because an interviewer usually expects you to explain your approach while you are writing up your solution. 


## Additional resources

* Daily plan for programming interview practice: https://github.com/jwasham/coding-interview-university#the-daily-plan


/wiki/programming/python-construct/
---
date: 2017-08-21
title: Python Construct Library
---

Sending data over serial communication channels in robotic systems that need real time updates between multiple agents calls for ensuring reliability and integrity. Construct is a Python library that comes to the rescue for building and parsing binary data efficiently without the need for writing unnecessary imperative code to build message packets.

The Python Construct is a powerful declarative parser (and builder) for binary data. It declaratively defines a data structure that describes the data. It can be used to parse data into python data structures or to convert these data structures into binary data to be sent over the channel. It is extremely easy to use after you install all the required dependencies into your machine.

## Features
Some key features of Construct are:
- Bit and byte granularity
- Easy to extend subclass system
- Fields: raw bytes or numerical types
- Structs and Sequences: combine simpler constructs into more complex ones
- Adapters: change how data is represented
- Arrays/Ranges: duplicate constructs
- Meta-constructs: use the context (history) to compute the size of data
- If/Switch: branch the computational path based on the context
- On-demand (lazy) parsing: read only what you require
- Pointers: jump from here to there in the data stream

You might not need to use all of the above features if the data you need to send and receive is a simple list of say, waypoints or agent IDs. But it is worth exploring the possible extent of complexity of data that this library can handle. The library provides both simple, atomic constructs (UBINT8, UBNIT16, etc), as well as composite ones which allow you form hierarchical structures of increasing complexity.

## Example Usage
This tool could especially come in handy if your data has many different kinds of fields. As an example, consider a message that contains the agent ID, message type (defined by a word, such as ‘Request’ or ‘Action’), flags and finally the data field that contains a list of integers. Building and parsing such a message would require many lines of code which can be avoided by simply using Construct and defining these fields in a custom format. An example message format is given below.

### Example format
```
message_crc = Struct('message_crc', UBInt64('crc'))

message_format = Struct('message_format',
    ULInt16('vcl_id'),
    Enum(Byte('message_type'),
    HELLO = 0x40,
    INTRO = 0x3f,
    UPDATE = 0x30,
    GOODBYE = 0x20,
    PARKED = 0x31,

        _default_ = Pass
    ),
    Byte('datalen'),
    Array(lambda ctx: ctx['datalen'], Byte('data')),
    Embed(message_crc)
)


if __name__ == "__main__":
    raw = message_format.build(Container(
    vcl_id=0x1,
        message_type='HELLO',
        datalen=4,
        data=[0x1, 0xff, 0xff, 0xdd],
        crc=0x12345678))

    print raw
    mymsg=raw.encode('hex')
    print mymsg
    x=message_format.parse(raw)
    print x
```

The CRC (Cyclic Redundancy Check) used in this snippet is an error-detecting code commonly used in digital networks to detect accidental changes to raw data. Blocks of data get a short check value attached usually at the end, based on the remainder of a polynomial division of their contents. On retrieval, the calculation is repeated and, in the event the check values do not match, exceptions can be handles in your code to take corrective action. This is a very efficient means to check validity of data received, which can be crucial to avoid errors in operation of real time systems.

There are a number of possible methods to check data integrity. CRC checks like CRC11, CRC12, CRC32, etc are commonly used error checking codes that can be used. If you face an issue with using CRC of varying lengths with your data, try using cryptographic hash functions like MD5, which might solve the problem. Further reading can be found [here on StackOverflow.](http://stackoverflow.com/questions/16122067/md5-vs-crc32-which-ones-better-for-common-use)

The snippets (python) below serve as examples of how to build and recover messages using Construct:
```
def build_msg(vcl_id, message_type):

    data = [0x1,0xff,0xff,0xdd]

    datalen = len(data)

    raw = message_format.build(Container(
        vcl_id = vcl_id,
        message_type = message_type,
        datalen = datalen,
        data = data,
        crc = 0))

    msg_without_crc = raw[:-8]
    msg_crc = message_crc.build(Container(
        crc = int(''.join([i for i in hashlib.md5(msg_without_crc).hexdigest() if i.isdigit()])[0:10])))

    msg = msg_without_crc + msg_crc

    pw = ProtocolWrapper(
            header = PROTOCOL_HEADER,
            footer = PROTOCOL_FOOTER,
            dle = PROTOCOL_DLE)
    return pw.wrap(msg)
```
```
def recover_msg(msg):
        pw = ProtocolWrapper(
        header = PROTOCOL_HEADER,
        footer = PROTOCOL_FOOTER,
        dle = PROTOCOL_DLE)

        status = map(pw.input, msg)
        rec_crc = 0
        calc_crc = 1

        if status[-1] == ProtocolStatus.MSG_OK:
                rec_msg = pw.last_message
                rec_crc = message_crc.parse(rec_msg[-8:]).crc
                calc_crc = int(''.join([i for i in hashlib.md5(rec_msg[:-8]).hexdigest() if i.isdigit()])[0:10])

        if rec_crc != calc_crc:
            print 'Error: CRC mismatch'
            return None
        else:
            return rec_msg
```

The following are a guidelines to use Python Construct:
1. Download the suitable version of Construct [here.](https://pypi.python.org/pypi/construct)
2. Copy over this folder to the appropriate code directory. If using ROS, it should be inside your ROS package)
3. You need two additional files `myformat.py` and `protocolwrapper.py` to get started. These can be found [here.](http://eli.thegreenplace.net/2009/08/20/frames-and-protocols-for-the-serial-port-in-python) This is a great resource for example code and supporting protocol wrappers to be used for serial communication (in Python)

## Resources
- Construct’s homepage is http://construct.readthedocs.org/ where you can find all kinds of docs and resources.
- The library itself is developed on https://github.com/construct/construct.
- For discussion and queries, here is a link to the [Google group](https://groups.google.com/forum/#!forum/construct3).
- Construct should run on any Python 2.5-3.3 implementation. Its only requirement is [six](http://pypi.python.org/pypi/six), which is used to overcome the differences between Python 2 and 3.

