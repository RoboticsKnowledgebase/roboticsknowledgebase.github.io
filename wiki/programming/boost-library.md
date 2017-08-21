---
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
