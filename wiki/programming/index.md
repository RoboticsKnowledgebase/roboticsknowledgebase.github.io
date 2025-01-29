---
date: 2024-12-05
title: Programming
---
<!-- **This page is a stub.** You can help us improve it by [editing it](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io).
{: .notice--warning} -->

This section focuses on **programming techniques, tools, and libraries** commonly used. It includes practical guides and examples for libraries like Boost, Eigen, and Python Construct, as well as programming concepts like multithreading, build systems, and preparing for technical interviews.

## Key Subsections and Highlights

- **[Boost C++ Libraries](/wiki/programming/boost-library/)**
  Highlights the widely-used Boost library in C++ for robotics. Discusses essential components like shared pointers, mutexes, and threading. Provides examples and resources for using Boost in robotics applications to minimize memory leaks and ensure thread safety.

- **[Boost: Iterations in Maps and Vectors](/wiki/programming/boost-maps-vectors/)**
  Explains how Boost simplifies working with maps and vectors in C++. Demonstrates an efficient approach to iterating through `std::map` using Boost's `FOREACH` and adaptors, with a practical example for managing robot IDs and poses.

- **[CMake and Other Build Systems](/wiki/programming/cmake/)**
  Provides an in-depth guide to CMake, a tool that simplifies building projects in compiled languages like C++ and integrates seamlessly with external libraries. Covers folder structure, common functions, and best practices for managing large projects and external dependencies.

- **[Eigen Geometry Library for C++](/wiki/programming/eigen-library/)**
  Details the use of Eigen for matrix and vector operations, transformations, and handling quaternions. Includes examples for creating rotation matrices, extracting translation vectors, and performing homogeneous transformations.

- **[Git](/wiki/programming/git/)**
  Covers Git as a distributed version control system. Explains GUI options, repository providers like GitHub, GitLab, and BitBucket, and includes learning resources from beginner to advanced levels. Offers tips on using Git effectively in robotics projects.

- **[Multithreaded Programming as an Alternative to ROS](/wiki/programming/multithreaded-programming/)**
  Discusses the use of pthreads and Boost for multithreading as a lightweight alternative to ROS for single-system applications. Includes a practical example for parallelizing IMU data processing and resources for learning pthreads.

- **[Programming Interviews](/wiki/programming/programming-interviews/)**
  A comprehensive guide for preparing for technical interviews in the robotics industry. Discusses algorithms, data structures, operating systems, and coding skills. Provides curated resources, LeetCode strategies, and system design tips for interview preparation.

- **[Python Construct Library](/wiki/programming/python-construct/)**
  Explores the Python Construct library for building and parsing binary data. Highlights its utility for reliable serial communication in robotics systems. Includes examples of creating structured messages with CRC error-checking.

## Resources

### General Programming Resources
- [Boost C++ Libraries Documentation](https://www.boost.org/)
- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- [Eigen Documentation](http://eigen.tuxfamily.org/dox/)
- [Git Documentation](http://git-scm.com/documentation)

### Technical Interview Preparation
- [LeetCode](https://leetcode.com/)
- [TopCoder Tutorials](http://www.topcoder.com/tc?module=Static&d1=tutorials&d2=alg_index)
- [Introduction to Algorithms by Cormen et al.](https://mitpress.mit.edu/books/introduction-algorithms)

### Multithreading and Binary Parsing
- [Pthreads Tutorial](https://computing.llnl.gov/tutorials/pthreads/)
- [Python Construct GitHub](https://github.com/construct/construct)
- [Protocol Wrappers for Python](http://eli.thegreenplace.net/2009/08/20/frames-and-protocols-for-the-serial-port-in-python)

### Tutorials and Forums
- [Learn Code the Hard Way](https://learncodethehardway.org/): Without a doubt, the best, most comprehensive tutorials for Python and C in existence.
- [CPlusPlus.com](https://www.cplusplus.com/): Website providing syntax, examples, and forums for C and C++ code for almost all of the standard library that should be on all \*nix systems.
- [StackOverflow](https://www.stackoverflow.com): Very popular and friendly programming community providing solutions to problems, and helpful tips for all programming languages.
- [TopCoder C++ Standard Templace Tutorial](https://www.topcoder.com/community/data-science/data-science-tutorials/power-up-c-with-the-standard-template-library-part-1/): Highlights the power of the C++ Standard library
- [LeetCode's Algorithm Problem Sets](https://leetcode.com/problemset/algorithms/): One the best places to practice for coding interviews.

### Books
- General programming best practices - all of these books are quick reads, and will save you tons of time in the long run
  - [Clean Code, by Robert C. Martin](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)
  - [Pragmatic Programmer, by Hunt and Thomas](https://www.amazon.com/Pragmatic-Programmer-Journeyman-Master/dp/020161622X)
  - [Debugging: The 9 Indispensable Rules, by David Agans](https://www.amazon.com/Debugging-Indispensable-Software-Hardware-Problems/dp/0814474578/)
- C++
  - [Effective C++, by Scott Meyers](https://www.amazon.com/Effective-Specific-Improve-Programs-Designs/dp/0321334876): A great second book on C++, going deeper into the language's constructs and best practices. Also check out his other books