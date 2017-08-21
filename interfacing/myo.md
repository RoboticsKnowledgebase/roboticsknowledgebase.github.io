---
title: Getting Started with the Myo
---
Here are some resources that are useful if you want to get started quickly with the [Myo Gesture Control Armband](https://www.myo.com/). The learning curve is not too steep if you want to make simple apps. It should take you just a couple of days (maybe even less) to get working. However, if you wish to work on more complex problems, the Lua environment used for Myo scripting will not be enough. If you have never worked with APIs before it will add more steepness to your learning curve. I have tried to find as many resources as possible and have structured them in the best way I could here. The lists here are by no means exhaustive but is meant to serve as a reference after you get acquainted with the basics.

## Lua Basics
The following links are meant to teach you everything from scratch. Myo uses the Lua scripting language which is quite easy to learn if you have worked with JavaScript before. If you have not I would suggest using one of the following links to make yourself familiar with Lua first (don't worry it's fairly easy):

Lua Tutorials
NOTE: List not exhaustive.

1. [The Lua Users Wiki](http://lua-users.org/wiki/LuaDirectory)
This site is very oddly structured but has some great content if you can figure how to find it.
2. [Lua Tutorial Series by TheCoolSquare](https://www.youtube.com/watch?v=dA9tcPeZa8k&list=PL5D2E7A4DD535E276)
Nothing teaches better than a video tutorial and these seem to be the most popular ones.
3. [The Official Lua Reference Manual](http://www.lua.org/manual/5.3/)
This by far is the best and the most exhaustive resource if you know the basics.

## Myo Basics
All of the following 6 links are available on the Myo's public forum but they are lost among hundreds of other blog posts. They are listed here along with a brief description of what you may expect to learn from each one of them.

1. [Setup and Getting Started](http://developerblog.myo.com/getting-started-myo-scripts-part-1/)
This takes you through the setup procedure and gets you started with the development environment.
2. [Programming with the API](http://developerblog.myo.com/getting-started-myo-scripts-part-2/)
If you have never worked with an API before, I hope this will ease you into it.
3. [Accessing the Data](http://developerblog.myo.com/getting-started-myo-scripts-part-3/)
This is how you start manipulating the gestures
4. [Keyboard Integration](http://developerblog.myo.com/getting-started-myo-scripts-part-4/)
Continuation of the API, but of limited value.
5. [Remaining Functions](http://developerblog.myo.com/getting-started-myo-scripts-part-5/)
This explains some of the functions that are not covered above.
6. [Combining Everything Together](http://developerblog.myo.com/getting-started-myo-scripts-part-6/)
Tutorial on combining the above components.

If you went through everything above while implementing it on your own, you are all set to make simple apps that can do anything from controlling your favorite music player to using your hand as a mouse. Now move on to the next part now.

## Advanced: Complete API Reference and Wrappers
If you are an experienced software programmer and have worked with APIs before (or you completed the basics section above) you can directly use [this for the complete API reference](https://developer.thalmic.com/docs/api_reference/platform/index.html). This not only contains all the API information but also has some demo programs to get you started. If you freak out at the prospect of writing hundreds of lines of code, I'd suggest downloading these samples and modifying them to build your applications. It's really fun that way.

You will soon realize that working with Lua is extremely useless as it does not give you the power to build bigger applications like Python or Java. For all those purposes, here is a list of all bindings in every language I could find. All of these are Github repositories and this is what will make your life easier and working with Myo real fun. There are many more out there somewhere but I have only listed the ones I have tried out myself and am sure worked properly at least at the time of writing this. I would suggest using Python if you have the option simply because I used it and most people I met in the forums used it too and also because it is really well written out. Some of these are by developers at Thalmic Labs. Most of the following contain all the examples available on the official website to help you get started. Again, if you don't want to write from scratch just modify the examples and build upon them.

- [Python](https://github.com/NiklasRosenstein/myo-python) Well documented
- [.JS](https://github.com/thalmiclabs/myo.js)
- [C#](https://github.com/tayfuzun/MyoSharp)
- [Unreal Engine](https://github.com/getnamo/myo-ue4)
- [Linux](https://github.com/freehaha/myo4l)
- [Java](https://github.com/NicholasAStuart/myo-java)
- [.NET](https://github.com/rtlayzell/Myo.Net)
- [Ruby](https://github.com/uetchy/myo-ruby)
- [ROS](https://github.com/roboTJ101/ros_myo)
