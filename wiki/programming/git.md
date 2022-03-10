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

- [Managing Merge Conflicts | GitHub Learning Lab](https://lab.github.com/githubtraining/managing-merge-conflicts)
- [Reviewing Pull Requests | GitHub Learning Lab](https://lab.github.com/githubtraining/reviewing-pull-requests)
- [GitHub Actions Basics | GitHub Learning Lab](https://lab.github.com/githubtraining/github-actions:-hello-world)
- [Cherry Picking Commits](https://www.atlassian.com/git/tutorials/cherry-pick)
- [Rebasing Branches](https://docs.github.com/en/get-started/using-git/about-git-rebase)

### Advanced
- [Reflog](https://www.atlassian.com/git/tutorials/rewriting-history/git-reflog)
- [Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Hooks, Enforcing Commit-Message Formats & ACLs](https://git-scm.com/book/en/v2/Customizing-Git-An-Example-Git-Enforced-Policy)
- [GitHub Action: Continuous Integration | GitHub Learning Lab](https://lab.github.com/githubtraining/github-actions:-continuous-integration)

## References & Help
- [Atlassian Git Cheat Sheet](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet): A handy set of commands to have around your desk for those quick command look ups.
- [How to Undo Almost Anything with Git](https://github.blog/2015-06-08-how-to-undo-almost-anything-with-git/): A blog post from GitHub that lists some scary scenarios in git and how to undo almost anything.
- [Dangit, Git!?](https://dangitgit.com/): Quick references for getting out of bad situations in git. Very powerful set of fixes, but doesn't provide a good explanation of what happened / how the fix works - read before you blindly follow the instructions.
- [Official Git Documentation](http://git-scm.com/documentation)
Extremely thorough documentation of git. It can be quite dense if you're new to git, but this the most authoritative and updated documentation for git CLI usage. Best used to look up command-line usage, parameters and their rationale.
- Man pages: The git command line interface ships with a very good set of man pages accessible by running `man git` on the terminal and is available wherever git is installed. If you're not familiar with man pages, you can read about it [here](https://itsfoss.com/linux-man-page-guide/).