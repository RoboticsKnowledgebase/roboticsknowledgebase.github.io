## Overview
The **Robotics Knowledgebase** is the Wiki for Robot Builders. It exists to advance knowledge in the robotics discipline. We document and share application details left out of textbooks and academic papers.

This Knowledgebase holds knowledge essential to building working robots. We believe robots should be built to solve specific problems. We take a systems-based approach to creating them. We include Project Management practices to ensure project completion.

## Contribution Quickstart Guide
### New Entries
To submit an original article follow these steps:
1. Fork this repository.
2. Locate the appropriate directory for your submission in `/wiki`.
3. Copy `_templates/template.md` into the directory.
4. Write your article. We recommend the following options:
  - Connect [Prose.io](http://prose.io/) to your Github account and write from your browser.
  - Clone the repository to your device. We recommend using the [Atom editor](https://atom.io/) with the  [Markdown-Writer](https://atom.io/packages/markdown-writer) and [Markdown-Image-Helper](https://atom.io/packages/markdown-image-helper) packages.
5. Rename `template.md` to the subject of your article. The new name should be lowercase, have hyphens for spaces, and end with `.md` (`this-is-an-example.md`)
6. Add a link to your article to `_data/navigation.yml` under the `wiki` heading.
  - Do not modify the `main` or `docs` lists.
7. Submit a pull request to the Robotics Knowledgebase.
  - If you're working from your device, don't forget to add, commit, and push your changes first.
8. Editors may request changes before they accept your pull request. Use their feedback to improve your entry and resubmit.

### Improving Articles
If you spot a mistake (or think that you have an improvement to an article), [create an issue](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io/issues) to discuss your recommended changes.

### Keeping your Fork Updated
Syncing a fork is accomplished through git on your local device. You should already have Git installed and cloned your fork to your computer.
1. Navigate to the working directory of your local project.
2. Configure a remote that points to the upstream repository. On your Linux device, use:
  - `git remote add upstream https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io`.
  - This only needs to be done once.
3. Use `git remote -v` to verify that the upstream repository is listed.
4. Fetch the latest commits from the upstream repository. These will be stored in a local branch `upstream/master`:
  - `git fetch upstream`
5. Check out your fork's local `master` branch:
  - `git checkout master`
6. This brings your fork's master branch into sync with the upstream repository, **without losing your local changes**:
  - `git merge upstream/master`

### Clear Changes and Update your Fork
This method is used specifically to erase all changes to your fork and replace it with the most updated copy of the Wiki.
1. Follow steps 1-5 from **Keeping your Fork Updated** above.
2. Reset your local repository:
  - `git reset --hard upstream/master `
3. Force the new repository to overwrite your remote fork:
  - `git push origin master --force`

## Site Structure
### Supporting technology
The Robotics Knowledgebase makes use of the following:
- [Github Pages](https://help.github.com/categories/20/articles)
- [Jekyll](https://jekyllrb.com/)
- [Minimal Mistakes Theme](https://mmistakes.github.io/minimal-mistakes/) by [Michael Rose](https://mademistakes.com/)
- [MathJax](http://docs.mathjax.org/en/latest/)

### Directories
The wiki itself is contained in the `/wiki` folder. The `/docs` folder exists to contain future documentation on contributing and supporting the wiki. The wiki contains the following categories:
- **Actuation**
  - Topics related to the moving components including motors and controls.
- **Common Platforms**
  - Covers common hardware and software frameworks used in robotics. ROS is included here.
- **Computing**
  - Topics related to hardware brains for robotics including on-board computers and cluster computing.
- **Fabrication**
  - Topics related to techniques and tools for fabricating a robot.
- **Interfacing**
  - Covers means of interfacing with a robot outside of networking.
- **Networking**
  - Topics related to communications infrastructure for robotics including programming and wireless technologies.
- **Programming**
  - General programming topics including languages and practices.
- **Project Management**
  - Topics related to project management practices.
- **Sensing**
  - Covers topics related to sensors including computer vision and cameras.
- **State Estimation**
  - Topics related to the position and orientation of a robot including navigation, localization, and mapping.
- **System Design and Development**
  - Covers topics related to Systems Engineering.
- **Tools**
  - Useful hardware and software for robotics not used directly in the robot itself.

## Directory Structure
Individual subfolders should contain both an `/assets` folder (for supporting files including images) and an `index.md` file.
## Future Work
This Knowledgebase is an evolving project. There are numerous areas for improvement in both content and site features.

### Needed topics
- Introduction to planning your robotics project
- Updated overview of single-board computers (`/wiki/computing/single-board-computers.md`)
- Mobility Overview (put in `/actuation`)
- Manipulation Overview (put in `/actuation`)
- State Estimation Overview
- System Engineering Overview
- Using GPUs for Computer Vision
- Using GPUs for Machine Learning
- V-model

### Todo
- Logo for Robotics Knowledgebase
- Writing & style standards
- Link to Github
- Collapsed Navigation for sidebar
- Separate repositories for content and technical files
- Dynamic Navigation Generation
- ~~Default template~~
- ~~Implement Math Support~~
