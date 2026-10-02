# Robotics Knowledgebase 🤖

[![Jekyll Build](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io/actions/workflows/jekyll-github-pages-deploy.yml/badge.svg)](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io/actions/workflows/jekyll-github-pages-deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The **Robotics Knowledgebase** is an open-source wiki for robot builders, researchers, and hobbyists. It focuses on the practical application details often omitted from textbooks—the "tribal knowledge" required to build, program, and deploy working robotic systems.

We take a **systems-based approach**, integrating engineering best practices with hands-on implementation guides for modern robotics frameworks.

---

## 🚀 Quickstart for Contributors

We welcome contributions from the community! Whether you're fixing a typo or adding a new deep-dive article, here's how to get started.

### Adding New Articles
1. **Fork** the repository and clone it locally.
2. **Choose a category** in the `/wiki` directory (e.g., `sensing`, `actuation`, `programming`).
3. **Template**: Copy `/_templates/template.md` to your chosen directory.
4. **Name**: Rename it using `kebab-case.md` (e.g., `my-new-sensor-guide.md`).
5. **Write**: Author your article using Markdown.
   - Use absolute paths for images: `![Alt Text](/assets/images/my_image.png)`.
   - Use absolute paths for internal links: `[Other Page](/wiki/category/other-page/)`.
6. **Navigation**: Add your entry to `/_data/navigation.yml` and the parent category's `index.md`.
7. **Build & Test**: See 'Local Development' below on instructions to build and deploylocally. Open all modified pages and verify your changes visually.
7. **PR**: Submit a Pull Request. Our editors will review and provide feedback. Follow through with changes until merge.

---

## 🛠 Local Development

To build and preview the site locally, we recommend using a Ruby virtual environment for dependency isolation.

### Prerequisites
- **Ruby version manager**: We recommend [rbenv](https://github.com/rbenv/rbenv) to manage Ruby versions.
  - **macOS**: `brew install rbenv`
  - **Linux**: See the [rbenv installer](https://github.com/rbenv/rbenv-installer) or use your package manager.

### Setup and Build
1. **Install Ruby**:
   From root of the project, run:
   ```bash
   rbenv install $(cat .ruby-version)
   ```
2. **Install Bundler**:
   ```bash
   gem install bundler
   ```
3. **Configure local path**:
   ```bash
   bundle config set --local path 'vendor/bundle'
   ```
4. **Install dependencies**:
   ```bash
   bundle install
   ```
5. **Local Preview**:
   ```bash
   bundle exec jekyll serve
   ```
   The site will be available at `http://localhost:4000`.

---

## 📂 Project Structure

### Supporting Technology
- **Engine**: [Jekyll](https://jekyllrb.com/) (Static Site Generator)
- **Theme**: [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/)
- **Hosting**: [GitHub Pages](https://pages.github.com/)
- **Math**: [MathJax](https://www.mathjax.org/) (LaTeX rendering)

### Key Directories
- `/wiki/`: The core content, organized by robotic subsystem.
- `/_data/`: Navigation and UI configuration.
- `/assets/images/`: Central store for all diagrams and photos.
- `/_templates/`: Base templates for new wiki entries.

---

## 🗺 Wiki Categories

- **[Controls & Actuation](/wiki/actuation/)**: Motors, PID, and motion control.
- **[Common Platforms](/wiki/common-platforms/)**: ROS, UAVs, and mobile bases.
- **[Computing](/wiki/computing/)**: SBCs, GPUs, and embedded controllers.
- **[Fabrication](/wiki/fabrication/)**: 3D printing, machining, and prototyping.
- **[Interfacing](/wiki/interfacing/)**: Microcontrollers and low-level protocols.
- **[Networking](/wiki/networking/)**: Communications and distributed systems.
- **[Programming](/wiki/programming/)**: Languages, libraries (Eigen, Boost), and practices.
- **[Project Management](/wiki/project-management/)**: Workflows for robotics teams.
- **[Sensing](/wiki/sensing/)**: Computer vision, LiDAR, and IMUs.
- **[State Estimation](/wiki/state-estimation/)**: SLAM, localization, and navigation.
- **[System Design](/wiki/system-design-development/)**: Systems engineering for robotics.
- **[Tools](/wiki/tools/)**: External software, editors, and utilities.

---

## 🤝 Community & Support
If you find a bug or have a suggestion, please [open an issue](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io/issues) or join the discussion in our Pull Requests.

---
*Maintained by the Robotics Knowledgebase Team. Built for the community, by the community.*
