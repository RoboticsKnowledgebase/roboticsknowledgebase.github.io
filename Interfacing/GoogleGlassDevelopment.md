The Painful Guide to Google Glass Development
NOTE: Unlike my Myo reference guide where I handpicked almost everything and validated it before posting I take no guaranty for the links on this page. Finding resources for the Glass is extremely hard and this is almost all there is that I know is not too outdated. At the time of writing this, it has been almost 5 months since Google stopped selling Google Glass and almost a year since Google stopped caring about the Google Glass.

First and foremost, ask yourself the very basic question: Are you really really really sure want to use Glass as a part of your project? I would suggest not making the mistakes I did when you search for an answer to that question. Here is something to help you out. I have listed down all possible problems I faced while developing on the Glass in that post and I will constantly keep referring back to it. So it might be a good idea to keep it opened in a different tab.

Okay now let's get down to business. Firstly, I would like you to know that the learning curve for Google Glass is very steep. If you are a novice at Glass but have good command over Java and have a bunch of Android development experience, you are good to go. If you know Java but haven't worked on Android Studio before it's better because you can directly learn with Glass without having to get accustomed to the variations in the development style when working with Android apps.

I begin by assuming that you are fluent in your Java skills (basic Java will not be enough).

Official Tools
Here is the official reference guide/tutorial/examples provided by Google. A word of caution: MOST OF IT IS HEAVILY OUTDATED.

Google Glass API, in its latest avatar works lets you develop either Immersions or Live Cards. Immersions are basically apps and live cards are continuously updating screens that are used for displaying fast changing content like a countdown.

Immersion Tutorial
Unfortunately, I have no non-video tutorials that can teach you how to make Live Cards. But if an Immersion is all you want to make then you are in luck because if you follow the link above, you will find perhaps the most complete and fully functional tutorial ever made for Google Glass. It is quite easy to understand even if you do not have a lot of Java experience as you can copy paste snippets of code and learn what's happening while doing that.

Here is a link to the Immersion app I made while when I first went through this tutorial (You might have to scroll down a bit): LINK

The Glass Class: The only proper video tutorials ever made
If you try searching for Google Glass tutorials on Google in all different ways possible, you are almost certainly never going to come across this extensive wealth of information about the Glass. You can even try searching on Youtube if you want and you will never find these. This link is a course website for a mini class taught on Google Glass Development at the HIT Lab in New Zealand. This contains all the information you will ever need (I think). You can find lecture slide decks, video recordings of the lectures and even the code handed out here. The only thing I would like to mention here is that I have not watched all the videos and hence I cannot comment on how outdated they are (going by their recording dates they might be a bit outdated) but I am sure they will bring you to a position from where you can start thinking on your own.

The Glass-Myo Code:
Yes, this exists. If you ever want to get your Myo hooked up to the Glass, you can follow this example and if all goes well, hopefully it will work. Only one thing to remember, you need the Myo's Android SDK for this.

ROS_GLASS_TOOLS
The reason I am putting this here is to let you know that THIS IS HEAVILY OUTDATED and warn you to STAY AWAY FROM THIS if you stumble upon it somehow. If you want to know why it is outdated, refer to my other post 
