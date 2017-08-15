---
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
