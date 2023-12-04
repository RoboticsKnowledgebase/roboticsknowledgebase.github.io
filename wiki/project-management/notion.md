---
date: {}
title: Using Notion for Project Management
published: true
---

Effective project management is crucial to the long-term success of any project, and there are various free tools that small teams (3-8 members) can use to effectively manage their project.

This document explains how to use one such tool, Notion, for Project Management. [Notion](https://www.notion.so/product) is a documentation and task-tracking platform that can be significantly useful in managing different aspects of a project. In the first section, this document details how to set up and use Notion for Project Management. The second section then introduces various project management strategies that teams can use to improve their project management process.

> The project manager’s role is to ensure that all aspects of the project are planned for and appropriately tracked throughout the life of the project. There should be at least one member in the team that assumes this role (could be in addition to other roles). 


## Setting up Notion for your team
Individual university students and teachers who associate their accounts with their school email addresses can get Notion’s Plus Plan for free. With this plan, one member in the team can host a teamspace which can be accessed and edited by upto 100 guests (Notion users). Here are the steps to set up Notion for your team: 

1. Create a Notion account with your educational email ID. 
2. Create a Notion team space. 
	- Inside your Notion workspace, click on the 'Settings and the members' option on the top-left of the window. 
    - Navigate to Teamspaces and create a new team space.  
3. Add team members to your new team space. 
	- Go to the home page of your team space.
    - On the top-right of the screen, click on the ‘Share’ button.
    -  Enter the email IDs of your team members and click ‘Invite’ to share the team space with them. Make sure the permissions allow invited members to access and edit the team space.
4. Each member invited should now have access to view and edit the team space. While this space is hosted on an individual’s Notion account, it is still a common team space which can be used by all members of the team (added as guests to the teamspace on Notion).

**Note:** The above method does not lead to a private teamspace on Notion. You will need to upgrade to Notion's premium plan in order to convert this to a private teamspace (only members can see the teamspace exists).





## Notion Project Management Template 
Here is a [simple Notion project management template](https://short-gatsby-680.notion.site/Dummy-Project-47645eb16177425d932332db325d0233?pvs=4) that you can use for your project. It contains the major sections that are needed to effectively manage your project. 
- To use the entire template in your project, just click on 'Duplicate' button on the top-right of the home page.
- To use a particular section of the template in your project, navigate to the section that you would like to use (for eg. Meeting Notes) and then click on the 'Duplicate' button on the top-right of the page. 
- When prompted where you would like to duplicate the tempalte to, just select your teamspace. 



## Meeting Notes




## T




This template acts as a tutorial on writing articles for the Robotics Knowledgebase. In it we will cover article structure, basic syntax, and other useful hints. Every tutorial and article should start with a proper introduction.

This goes above the first subheading. The first 100 words are used as an excerpt on the Wiki's Index. No images, HTML, or special formating should be used in this section as it won't be displayed properly.

If you're writing a tutorial, use this section to specify what the reader will be able to accomplish and the tools you will be using. If you're writing an article, this section should be used to encapsulate the topic covered. Use Wikipedia for inspiration on how to write a proper introduction to a topic.

In both cases, tell them what you're going to say, use the sections below to say it, then summarize at the end (with suggestions for further study).

## First subheading
Use this section to cover important terms and information useful to completing the tutorial or understanding the topic addressed. Don't be afraid to include to other wiki entries that would be useful for what you intend to cover. Notice that there are two \#'s used for subheadings; that's the minimum. Each additional sublevel will have an added \#. It's strongly recommended that you create and work from an outline.

This section covers the basic syntax and some rules of thumb for writing.

### Basic syntax
A line in between create a separate paragraph. *This is italicized.* **This is bold.** Here is [a link](/). If you want to display the URL, you can do it like this <http://ri.cmu.edu/>.

> This is a note. Use it to reinforce important points, especially potential show stoppers for your readers. It is also appropriate to use for long quotes from other texts.


#### Bullet points and numbered lists
Here are some hints on writing (in no particular order):
- Focus on application knowledge.
  - Write tutorials to achieve a specific outcome.
  - Relay theory in an intuitive way (especially if you initially struggled).
    - It is likely that others are confused in the same way you were. They will benefit from your perspective.
  - You do not need to be an expert to produce useful content.
  - Document procedures as you learn them. You or others may refine them later.
- Use a professional tone.
  - Be non-partisan.
    - Characterize technology and practices in a way that assists the reader to make intelligent decisions.
    - When in doubt, use the SVOR (Strengths, Vulnerabilities, Opportunities, and Risks) framework.
  - Personal opinions have no place in the Wiki. Do not use "I." Only use "we" when referring to the contributors and editors of the Robotics Knowledgebase. You may "you" when giving instructions in tutorials.
- Use American English (for now).
  - We made add support for other languages in the future.
- The Robotics Knowledgebase is still evolving. We are using Jekyll and GitHub Pages in and a novel way and are always looking for contributors' input.

Entries in the Wiki should follow this format:
1. Excerpt introducing the entry's contents.
  - Be sure to specify if it is a tutorial or an article.
  - Remember that the first 100 words get used else where. A well written excerpt ensures that your entry gets read.
2. The content of your entry.
3. Summary.
4. See Also Links (relevant articles in the Wiki).
5. Further Reading (relevant articles on other sites).
6. References.

#### Code snippets
There's also a lot of support for displaying code. You can do it inline like `this`. You should also use the inline code syntax for `filenames` and `ROS_node_names`.

Larger chunks of code should use this format:
```
def recover_msg(msg):

        // Good coders comment their code for others.

        pw = ProtocolWrapper()

        // Explanation.

        if rec_crc != calc_crc:
            return None
```
This would be a good spot further explain you code snippet. Break it down for the user so they understand what is going on.

#### LaTex Math Support
Here is an example MathJax inline rendering $ \phi(x\|y) $ (note the additional escape for using \|), and here is a block rendering:
$$ \frac{1}{n^{2}} $$

#### Images and Video
Images and embedded video are supported.

![Put a relevant caption here](assets/images/Hk47portrait-298x300.jpg)

{% include video id="8P9geWwi9e0" provider="youtube" %}

{% include video id="148982525" provider="vimeo" %}

The video id can be found at the end of the URL. In this case, the URLs were
`https://www.youtube.com/watch?v=8P9geWwi9e0`
& `https://vimeo.com/148982525`.

## Summary
Use this space to reinforce key points and to suggest next steps for your readers.

## See Also:
- Links to relevant material within the Robotics Knowledgebase go here.

## Further Reading
- Links to articles of interest outside the Wiki (that are not references) go here.

## References
- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your published report and link to it here.
