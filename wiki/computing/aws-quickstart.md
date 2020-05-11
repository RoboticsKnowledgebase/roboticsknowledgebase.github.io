---
title: Amazon Web Services Quickstart
---

This article will cover the basics of remote login on an Ubuntu machine. More specifically this will help you set up your AWS machine and serve as a tutorial to launch, access and manage your AWS Instance.
## Launching an EC2 Instance
First, we will have to sign up on AWS.
After logging into your account you will have to Choose a region.
The instances you make are linked to specific regions. After you have selected your region, click on Services in the top left. Then select EC2 under Compute.

1.  Launch an EC2 instance (Click on Launch Instance on the Dashboard)
2.  Select your required AMI. This will create a virtual machine for you with some pre-installed packages/applications. We will use the Deep Learning Base AMI for our tutorial.
    An Amazon Machine Image (AMI) provides the information required to launch an instance. You must specify an AMI when you launch an instance. You can launch multiple instances from a single AMI when you need multiple instances with the same configuration. You can use different AMIs to launch instances when you need instances with different configurations. -source AWS Website
3.  Select the version that matches your operating system (Ubuntu 16.04 / Ubuntu 18.04)
4.  Select the instance type. If you created a new account on AWS, you will be eligible for free usage of machines in the free tier range. T2.micro which falls under this category can be used to get familiar with AWS. To know more about the free-tier visit <https://aws.amazon.com/free/free-tier-faqs/>
5.  After selecting your instance, click configure Instance and dd torage.
6.  Depending on your requirement, select the amount of storage required. The first 30GB of storage is free under the one year eligibility.
7.  Continue pressing next until you see ‘Security Groups’.
    Here you will define the open ports for your machine. By default, port 22 will be open for you to ssh into your machine. However, you will have to define certain rules if you want to host different applications.
8.  Continue pressing ext until you see the review page.
9.  Launch Instance and select keypair. If you have previously generated a keypair, you can use the same file to access different machines.
10. To check your instances, click on Instances in the left sidebar.


**Now, to login to your instance -**
1.  Go to your terminal and login to your instance with the following command:\
    `ssh -i keyPair.pem -L 8000:localhost:8888 ubuntu@instance`
2.  You may need to set permissions for your key file. This can be done using chmod 400 keyPair.pem

Instead of having to write this huge command in your terminal, you can edit your ssh config file and use an alias to log in to your remote machine. For mac users the config file can be found in **/Users/user_name/.ssh/config**

The file should include\
*Host alias*\
*HostName remote_machine_ip*\
*IdentityFile path_to_keyfile/keyPair.pem*\
*User ubuntu*

## Using TMUX / SCREEN
While using ssh for remote access, your connections may be terminated if there is no activity for a long period of time. While training large models, this may be a problem. As your connection is piped through ssh, once the machine is left idle, the connection breaks causing all running applications / processes to terminate. To avoid this we can use Tmux or Screen.

TMUX​ is a ​terminal multiplexer​ for ​Unix-like​ ​operating systems​. It allows multiple ​terminal sessions to be accessed simultaneously in a single window. It is useful for running more than one ​command-line​ program at the same time. It can also be used to detach ​processes from their controlling terminals, allowing ​SSH​ sessions to remain active without being visible.
*- Wikipedia*

Here are a few important references in this regard
1.  [TMUX Quickstart](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/)
2.  [TMUX-Cheatsheet](https://tmuxcheatsheet.com/)

## Stopping Instances
To stop working with your instance for the night (or an extended period of time),
1. Go to your running instances
2. Select your active instance
3. Select Actions --> Instance State --> Stop

When your instance is not active, you will only be charged for storage (which is fairly cheap, but could add up.) To start the instance back up, follow the same steps but select start.
I you are done with an instance, follow the same steps, but terminate instead of stopping the instance.


## Spot Instances

A Spot Instance is an unused EC2 instance that is available for less than the On-Demand price. Because Spot Instances enable you to request unused EC2 instances at steep discounts, you can lower your Amazon EC2 costs significantly. The hourly price for a Spot Instance is called a Spot price. The Spot price of each instance type in each Availability Zone is set by Amazon EC2, and adjusted gradually based on the long-term supply of and demand for Spot Instances. Your Spot Instance runs whenever capacity is available and the maximum price per hour for your request exceeds the Spot price.

Spot Instances are a cost-effective choice if you can be flexible about when your applications run and if your applications can be interrupted. For example, Spot Instances are well-suited for data analysis, batch jobs, background processing, and optional tasks. For more information, see Amazon EC2 Spot Instances - AWS Official

Check how to use spot instances over here:
[Using Spot Instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html#spot-get-started)

## A word on Visual Studio Code:
VsCode comes with a plugin for remotely logging into your machine. This way you can develop and edit your code using VS Code-
The Visual Studio Code Remote - SSH extension allows you to open a remote folder on any remote machine, virtual machine, or container with a running SSH server and take full advantage of VS Code's feature set. Once connected to a server, you can interact with files and folders anywhere on the remote filesystem.
No source code needs to be on your local machine to gain these benefits since the extension runs commands and other extensions directly on the remote machine.

The following link provides details on the same.\
[Visual Studio - Remote SSH](https://code.visualstudio.com/blogs/2019/07/25/remote-ssh)
