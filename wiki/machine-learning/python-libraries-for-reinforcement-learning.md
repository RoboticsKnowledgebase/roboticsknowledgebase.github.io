---
date: 2020-12-07
title: Python libraries for Reinforcement Learning
---
**Reinforcement Learning (RL)** is a machine learning approach for teaching agents how to solve tasks by trial and error. More specifically, RL is mostly concerned with how software agents should take actions in an environment in order to maximize its cumulative reward. The application of RL, as it seeks a solution to balance exploration and exploitation, ranges from Resource Management, Traffic Light Control, Recommendation, and Advertising, to Robotics. The successes of deep learning and reinforcement learning area in recent years have led many researchers to develop methods to control robots using RL with the motivation to automate the process of designing sensing, planning, and control algorithms by letting the robot learn them autonomously. This post gives a brief introduction to a few popular RL libraries in the Robotics context, from beginning to immediate, to advanced level users. At last, we will provide general tips for a more in-depth study on RL topics and link to a concrete example of using RL to formulate a self-driving agent.

## [Spinning up](https://spinningup.openai.com/en/latest/)

If you are new to RL, [Spinning up](https://spinningup.openai.com/en/latest/) will provide you a comfortable jumpstart to get you started. As part of a new education initiative at OpenAI, Spinning up gives the formula to learn RL from scratch with the following core components:
- A short introduction to RL terminology, kinds of algorithms, and basic theory.
- A curated list of important papers organized by topic to familiarize with RL concepts on Model-Free RL, Model-based RL, and safe RL.
- Well-documented, short, standalone implementations of Vanilla Policy Gradient (VPG), Trust Region Policy Optimization (TRPO), Proximal Policy Optimization (PPO), Deep Deterministic Policy Gradient (DDPG), Twin Delayed DDPG (TD3), and Soft Actor-Critic (SAC).
- A few exercises to serve as warm-ups.
- A list of challenges and requests in terms of RL research topics.

### Hello World with Spinning Up

The best way to get a feel for how deep RL algorithms perform is to just run them. We provide a Hello World running [Proximal Policy Optimization (PPO)](https://openai.com/blog/openai-baselines-ppo/#ppo) With Spinning Up, that’s as easy as:
```
python -m spinup.run ppo --env CartPole-v1 --exp_name hello_world
```
and it hints at the standard way to run any Spinning Up algorithm from the command line:
```
python -m spinup.run [algo name] [experiment flags]
```
And you could also specify if you want to use a PyTorch version or Tensorflow version of an algorithm, just run with 
```
python -m spinup.run [algo]_pytorch [experiment flags]
```
or 
```
python -m spinup.run [algo]_tf1 [experiment flags]
```
Otherwise, the runner will look in  `spinup/user_config.py` for which version it should default to for that algorithm.
>If you are using ZShell: ZShell interprets square brackets as special characters. Spinning Up uses square brackets in a few ways for command line arguments; make sure to escape them, or try the solution recommended here if you want to escape them by default.

one could find more details about PPO from the OpenAI baseline [here](https://openai.com/blog/openai-baselines-ppo/).


## [Stable Baseline](https://github.com/hill-a/stable-baselines)

After a deep familiarization with RL concepts and learning from standard implementations of typical RL algorithms, one may refer to [Stable Baseline](https://github.com/hill-a/stable-baselines) for a set of improved implementations of RL algorithms. Stable Baseline is an extension to OpenAI [Baselines](https://github.com/openai/baselines), with:
- A collection of pre-trained agents with <https://github.com/araffin/rl-baselines-zoo>
- Unified structure for all algorithms
- PEP8 compliant (unified code style)
- Documented functions and classes
- More tests & more code coverage
- Additional algorithms: SAC and TD3 (+ HER support for DQN, DDPG, SAC, and TD3).

One may find himself or herself confused with the plethora of algorithms provided. As for which algorithm to use, Stable Baseline provides detailed instruction on choosing the provided algorithms with narrowing down the actions to be discrete or continuous, whether you can parallelize your training or not, and how you intend to achieve that ..., and provides a few general tips to research work related to RL:
- Read about RL and Stable Baselines
- Do quantitative experiments and hyperparameter tuning if needed
- Evaluate the performance using a separate test environment
- For better performance, increase the training budget
more details could be found in this [article](https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html).

### Hello World with Stable Baseline
Please follow the instructions [here](https://stable-baselines.readthedocs.io/en/master/guide/install.html) to install Stable Baseline with the appropriate systems.

>Stable-Baselines supports Tensorflow versions from 1.8.0 to 1.15.0, and does not work on Tensorflow versions 2.0.0 and above. PyTorch support is done in Stable-Baselines3

With Stable Baselines, training a PPO agent is as simple as:
```
from stable_baselines import PPO2

# Define and train a model in one line of code !
trained_model = PPO2('MlpPolicy', 'CartPole-v1').learn(total_timesteps=10000)
# you can then access the gym env using trained_model.get_env()
```
And Stable Baselines provides a [Colab Notebook](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/stable_baselines_getting_started.ipynb) for illustration.


## [RLlib](https://docs.ray.io/en/master/rllib.html)

If one is looking for a Fast and Parallel RL platform, Ray and RLlib would be the go-to. Ray is more than just a library for multi-processing; Ray’s real power comes from the RLlib and Tune libraries that leverage this capability for reinforcement learning. It enables you to scale training to large-scaled distributed servers, or just take advantage of the parallelization properties to more efficiently train using your own laptop.

RLlib, then, serves as an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications. RLlib natively supports TensorFlow, TensorFlow Eager, and PyTorch, but most of its internals are framework agnostic. An overview of RLlib's architecture could be illustrated with the graph here:
![RLlib Architecture](https://docs.ray.io/en/master/_images/rllib-stack.svg)

To get started, take a look over the [custom env example](https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py) and the API [documentation](https://docs.ray.io/en/master/rllib-toc.html). If you want to develop custom algorithms with RLlib, RLlib also provides detailed [instructions](https://docs.ray.io/en/master/rllib-concepts.html) to do so.

### Hello World with RLlib

RLlib has extra dependencies on top of `ray`. First, you’ll need to install either `PyTorch` or `TensorFlow`. Then, install the RLlib module:
```
pip install 'ray[rllib]'  # also recommended: ray[debug]
pip install gym
```
Then, you could have your first RL agent working with a standard, OpenAI Gym environment:
```
import ray
from ray.rllib import agents
ray.init() # Skip or set to ignore if already called
config = {'gamma': 0.9,
          'lr': 1e-2,
          'num_workers': 4,
          'train_batch_size': 1000,
          'model': {
              'fcnet_hiddens': [128, 128]
          }}
trainer = agents.ppo.PPOTrainer(env='CartPole-v0', config=config)
results = trainer.train()
```
The `config` dictionary is the configuration file, which details the setup to influence the number of layers and nodes in the network by nesting a dictionary called a model in the config dictionary. Once you have specified our configuration, calling the train() method on the trainer object will send the environment to the workers and begin collecting data. Once enough data is collected (1,000 samples according to the example settings above) the model will update and send the output to a new dictionary called results.

## Summary
To summarize, we provide a short introduction to three of the popular RL libraries in this post, while **Spinning Up** provides a friendly walk-through of the core RL concepts along with examples, **Stable Baselines** provides an efficient implementation to most of the popular RL algorithms. On the other hand, 
RLlib offers scalability. Note there is no silver bullet in RL, depending on your needs and problem, you may choose one or the other platform, or algorithm. But if you decide to have your own implementations instead of using the library, we recommend the following tips:
- Read the original paper several times
- Read existing implementations (if available)
- Validate the implementation by making it run on harder and harder ends (you can compare results against the RL zoo) and Always run hyperparameter optimization


## See Also:
- A survey on RL with Robotics could be found [here](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Kober_IJRR_2013.pdf).
- Applying RL Algorithms for [real world problems](https://towardsdatascience.com/applications-of-reinforcement-learning-in-real-world-1a94955bcd12) and  [Robotics field](https://towardsdatascience.com/reinforcement-learning-for-real-world-robotics-148c81dbdcff).
- A concrete [project](https://mrsdprojects.ri.cmu.edu/2020teamd/) of formulating an RL-driven self-driving agent in Simulation for safety.

## References
- J. Achiam, “Spinning Up in Deep RL,” OpenAI, 02-Sep-2020. [Online]. Available: https://openai.com/blog/spinning-up-in-deep-rl/.
- Hill-A, “hill-a/stable-baselines,” GitHub. [Online]. Available: https://github.com/hill-a/stable-baselines. 
- Ray-Project, “ray-project/ray.” [Online]. Available: https://github.com/ray-project/ray.
