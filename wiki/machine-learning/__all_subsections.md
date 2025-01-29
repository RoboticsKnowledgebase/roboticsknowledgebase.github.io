/wiki/machine-learning/custom-semantic-data/
---
date: 2020-04-10
title: Custom data-set for segmentation
published: true
---
Image segmentation can be useful in a lot of cases, for example, suppressing pedestrains, cars for your SLAM system so the extracted features belong to static scene only. This tutorial covers the procedure to create annotations for semantic segmentation task. This is useful when you want to outsource the labeling tasks to external companies because guidelines and examples are usually required in such scenario. Specifically, GIMP (GNU Image Manipulation Program version 2.8.22) on Ubuntu (16.04 for this tutorial) will be used to do the annotating task.

## Example of segmented images
Below is an example of annotated image and it's original RGB image. Three classes: fungus (annotated red), holes (annotated green) and background (annotated black) are presented here. Although it's common to use gray image with pixel value corresponding to 0 - 255, using a color annotation makes it much easier to visualize the annoation. The conversion from color to class labels can be easily done when the actual training is performed e.g. a mapping from RGB tuple to integers.

### Example code for converting RGB tuple to integer
```
color_of_interest = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0)]
class_map = dict(zip(self.color_of_interest, range(3)))

def encode_segmap(self, mask):
	for color in self.color_of_interest:
    	mask[ (mask[:,:,0] == color[0]) &\
        	(mask[:,:,1] == color[1]) &\
        	(mask[:,:,2] == color[2])   ] = class_map[color]
    return mask[:, :, 0] # target should be h x w, no depth
```

![mask annotation](assets/mask_annotation.png)

## Installing gimp
copy and paste the following command in your terminal to install gimp  
```
sudo apt-get update
sudo apt-get install gimp
```

## Procedure to annotate an image
### Step 1 Load image
Navigate to  file->open botton on the top left to open a rgb image that you'd like to annotate

### Step 2 Create mask
Navigate to layer->new layer to create a mask over your image. Choose Foreground color will create a black layer over your image. You can also change the foreground color on the left panel before you create a new layer, this will give you a layer with different color (which would corresponds to background in this tutorial)

![mask annotation](assets/new_layer.png)

After creating new layer, you will see your newly created layer on the right panel. Click on the eye symbol and make the layer invisible.

![manage layers](assets/manage_layers.png)

### Step 3 Creating annotations
Select the free select tool on the left panel. **IMPORTANT:** Uncheck the anti-aliasing option, otherwise non-solid colors will appear at the edge of your annotations. Select the region of interest, and then use bucket fill tool to fill in color annotation. Click on the eye symbol again on the right panel will show you the annotated layer.

![free select tool](assets/select_tool.png)
![create annotation](assets/bucket_fill.png)

### Step 4 Saving files
Hit ctrl+E to export your layer as an png image, which is your label for this image. Hit ctrl+S to save the gimp file as .xcf file. This step is important if you want to modify your annotation in the future.

## See Also:
- Semantic segmented images are sufficient for many architectures e.g. Unet, but if you'd like to work with Mask-RCNN, a .json file is required for training. Here's a [decent tutorial](http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)


/wiki/machine-learning/intro-to-rl/
---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2020-12-06 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Introduction to Reinforcement Learning
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.

---
The goal of Reinforcement Learning (RL) is to learn a good strategy for the agent from experimental trials and relative simple feedback received. With the optimal strategy, the agent is capable to actively adapt to the environment to maximize future rewards.

## Key Concepts

### Bellman Equations

Bellman equations refer to a set of equations that decompose the value function into the immediate reward plus the discounted future values.
$$\begin{aligned}
  V(s) &= \mathbb{E}[G_t | S_t = s]\\
   &= \mathbb{E}[R_{t+1} + \gamma V(S_{t+1})|S_t = s)]\\
 Q(s,a)&=\mathbb{E}[R_{t+1} + \gamma V(S_{t+1}) | S_t = s, A_t = a]
\end{aligned}$$

#### Bellman Expectation Equations

$$\begin{aligned}
  V_{\pi}(s) &= \sum_a \pi(a|s)\sum_{s',r} p(s', r | s, a)[r + \gamma V_{\pi}(s')]\\
  Q_\pi(s, a) &= \sum_{s'}\sum_{r}p(s', r | s, a)[r +\gamma\sum_{a'}\pi(a', s')Q_\pi(s', a')]
\end{aligned}
$$

#### Bellman Optimality Equations

$$\begin{aligned}
  V_*(s) &= \max_{a}\sum_{s'}\sum_{r}p(s', r | s, a)[r + \gamma V_*(s')]\\
  Q_*(s,a) &= \sum_{s'}\sum_{r}p(s', r | s, a)[r +\gamma\max_{a'}Q_*(s', a')]
\end{aligned} $$

## Approaches

### Dynamic Programming

When the model of the environment is known, following Bellman equations, we can use Dynamic Programming (DP) to iteratively evaluate value functions and improve policy.

#### Policy Evaluation

$$
V_{t+1} = \mathbb{E}[r+\gamma V_t(s') | S_t = s] = \sum_a\pi(a|s)\sum_{s', r}p(s', r|s,a)(r+\gamma V_t(s'))
$$

#### Policy Improvement

Given a policy and its value function, we can easily evaluate a change in the policy at a single state to a particular action. It is a natural extension to consider changes at all states and to all possible actions, selecting at each state the action that appears best according to $q_{\pi}(s,a).$ In other words, we make a new policy by acting greedily.

$$
Q_\pi(s, a) = \mathbb{E}[R_{t+1} + \gamma V_\pi(S_{t+1}) | S_t = s, A_t = a] = \sum_{s', r} p(s', r|s, a)(r+\gamma V_\pi (s'))
$$

#### Policy Iteration

Once a policy, $\pi$, has been improved using $V_{\pi}$ to yield a better policy, $\pi'$, we can then compute $V_{\pi}'$ and improve it again to yield an even better $\pi''$. We can thus obtain a sequence of monotonically improving policies and value functions:
$$\pi_0 \xrightarrow{E}V_{\pi_0}\xrightarrow{I}\pi_1 \xrightarrow{E}V_{\pi_1}\xrightarrow{I}\pi_2 \xrightarrow{E}\dots\xrightarrow{I}\pi_*\xrightarrow{E}V_{\pi_*}$$
where $\xrightarrow{E}$ denotes a policy evaluation and $\xrightarrow{I}$ denotes a policy improvement.

### Monte-Carlo Methods
Monte-Carlo (MC) methods require only experience --- sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. It learns from actual experience without no prior knowledge of the environment's dynamics. To compute the empirical return $G_t$, MC methods need to learn complete episodes $S_1, A_1, R_2, \dots, S_T$ to compute $G_t = \sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}$ and all the episodes must eventually terminate no matter what actions are selected.

The empirical mean return for state $s$ is:
$$V(s)=\frac{\sum_{t=1}^T\mathbf{1}[S_t=s]G_t}{\sum_{t=1}^T\mathbf{1}[S_t = s]}$$
Each occurrence of state $s$ in an episode is called a visit to $s$. We may count the visit of state $s$ every time so that there could exist multiple visits of one state in one episode ("every-visit"), or only count it the first time we encounter a state in one episode ("first-visit"). In practical, first-visit MC converges faster with lower average root mean squared error. A intuitive explanation is that it ignores data from other visits to $s$ after the first, which breaks the correlation between data resulting in unbiased estimate. 

This way of approximation can be easily extended to action-value functions by counting $(s, a)$ pair.
$$Q(s,a) = \frac{\sum_{t=1}^T\mathbf{1}[S_t = s, A_t = a]G_t}{\sum_{t=1}^T[S_t = s, A_t =a]}$$
To learn the optimal policy by MC, we iterate it by following a similar idea to Generalized Policy iteration (GPI).

1. Improve the policy greedily with respect to the current value function: $$\pi(s) = \arg\max_{a\in A}Q(s,a)$$

2. Generate a new episode with the new policy $\pi$ (i.e. using algorithms like $\epsilon$-greedy helps us balance between exploitation and exploration)

3. Estimate $Q$ using the new episode: $$q_\pi(s, a) = \frac{\sum_{t = 1}^T(\mathbf{1}[S_t = s, A_t = a]\sum_{k = 0}^{T-t-1}\gamma^kR_{t+k+1})}{\sum_{t=1}^T\mathbf{1}[S_t = s, A_t = a]}$$

### Temporal-Difference Learning

Temporal-difference (TD) learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo methods, TD methods can learn from raw experience without a model of the environment's dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).
Similar to Monte-Carlo methods, Temporal-Difference (TD) Learning is model-free and learns from episodes of experience. However, TD learning can learn from incomplete episodes.
$$Q(s, a) = R(s,a) + \gamma Q^\pi(s',a')$$

#### Comparison between MC and TD}

MC regresses $Q(s,a)$ with targets $y = \sum_i r(s_i, a_i)$. Each rollout has randomness due to stochasticity in policy and environment. Therefore, to estimate $Q(s,a)$, we need to generate many trajectories and average over such stochasticity, which is a high variance estimate. But it is unbiased meaning the return is the true target.

TD estimates $Q(s,s)$ with $y = r(s,a)+\gamma Q^\pi(s',a')$, where $Q^\pi(s',a')$ already accounts for stochasticity of future states and actions. Thus, the estimate has lower variance meaning it needs fewer samples to get a good estimate. But the estimate is biased: if $Q(s', a')$ has approximation errors, the target $y$ has approximation errors; this could lead to unstable training due to error propagation.

#### Bootstrapping

TD learning methods update targets in the following equation with regard to existing estimates rather than exclusively relying on actual rewards and complete returns as in MC methods in Equation (\ref{eq:9}). This approach is known as bootstrapping.
$$
\begin{aligned}
V(S_t) &\leftarrow V(S_t) +\alpha[G_t - V(S_t)]\\
V(S_t) &\leftarrow V(S_t) +\alpha[R_{t+1} +\gamma V(S_t) - V(S_t)]
\end{aligned}
$$

## Summary

Here are some simple methods used in Reinforcement Learning. There are a lot of fancy stuff, but due to limited pages, not included here. Feel free to update the wiki to keep track of the latest algorithms of RL.

## See Also:

<https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html>

## Further Reading

- Introduction to Reinforcement Learning, MIT Press

## References

Kaelbling, Leslie Pack, Michael L. Littman, and Andrew W. Moore. "Reinforcement learning: A survey." Journal of artificial intelligence research 4 (1996): 237-285.

/wiki/machine-learning/mediapipe-live-ml-anywhere/
---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2022-05-02 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Mediapipe - Live ML Anywhere
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

## Introduction - What is Mediapipe?

MediaPipe offers cross-platform, customizable ML solutions for live and streaming media. With common hardware, Mediapipe allows fast ML inference and processing. With Mediapipe, you can deploy the solutions anywhere, on Android, iOS, desktop/cloud, web and IoT platforms. The advantage of Mediapipe is that you get cutting-edge ML solutions that are free and open source. 

## Solutions offered

![Figure 1. Mediapipe Solutions](../assets/mediapipe_solutions.png)
The image above summarizes the solutions offered by mediapipe. 
The solutions below have been classified into 2 categories based on the use cases:

Following are the solutions offered for the detection of humans and their body parts: 
1. Face detection
MediaPipe Face Detection is an ultra-fast face detection solution that comes with 6 landmarks and multi-face support. 

2. FaceMesh
MediaPipe Face Mesh is a solution that estimates 468 3D face landmarks in real-time even on mobile devices. 

3. Mediapipe Hands
MediaPipe Hands is a high-fidelity hand and finger tracking solution. It employs machine learning (ML) to infer 21 3D landmarks of a hand from just a single frame.

4. MediaPipe Pose
MediaPipe Pose is an ML solution for high-fidelity body pose tracking, inferring 33 3D landmarks and background segmentation mask on the whole body from RGB video frames. 

5. MediaPipe Holistic
The MediaPipe Holistic pipeline integrates separate models for the pose, face and hand components, each of which is optimized for its particular domain.

6. MediaPipe Hair Segmentation
MediaPipe Hair Segmentation segments the hairs on the human face. 

7. MediaPipe Selfie Segmentation
MediaPipe Selfie Segmentation segments the prominent humans in the scene. It can run in real-time on both smartphones and laptops. 

Following are the solutions offered for the detection and tracking of everyday objects
1. Box tracking 
The box tracking solution consumes image frames from a video or camera stream, and starts box positions with timestamps, indicating 2D regions of interest to track, and computes the tracked box positions for each frame.

2. Instant Motion tracking
MediaPipe Instant Motion Tracking provides AR tracking across devices and platforms without initialization or calibration. It is built upon the MediaPipe Box Tracking solution. With Instant Motion Tracking, you can easily place virtual 2D and 3D content on static or moving surfaces, allowing them to seamlessly interact with the real-world environment.

3. Objectron
MediaPipe Objectron is a mobile real-time 3D object detection solution for everyday objects. It detects objects in 2D images, and estimates their poses through a machine learning (ML) model, trained on the Objectron dataset.

4. KNIFT  
MediaPipe KNIFT is a template-based feature matching solution using KNIFT (Keypoint Neural Invariant Feature Transform). KNIFT is a strong feature descriptor robust not only to affine distortions, but to some degree of perspective distortions as well. This can be a crucial building block to establish reliable correspondences between different views of an object or scene, forming the foundation for approaches like template matching, image retrieval and structure from motion.

The table below describes the support of the above models for currently available platforms:
![Figure 1. Mediapipe supported platforms](../assets/mediapipe_platforms.png)

## Quickstart Guide
Mediapipe solutions are available for various platforms viz. Android, iOS, Python, JavaScript, C++. The guide at [Getting Started](https://google.github.io/mediapipe/getting_started/getting_started.html) comprises instructions for various platforms. 

For this section of the quick-start guide, we will introduce you to getting started using Python.
MediaPipe offers ready-to-use yet customizable Python solutions as a prebuilt Python package. MediaPipe Python package is available on [PyPI](https://pypi.org/project/mediapipe/) for Linux, macOS and Windows.

1. Step 1 - Activate the virtual environment:

```
python3 -m venv mp_env && source mp_env/bin/activate
```

The above code snippet will create a virtual environment `mp_env` and start the virtual environment. 

2. Step 2 - Install MediaPipe Python package using the following command:

```
(mp_env)$ pip3 install mediapipe
```

You are all set! You can now start using mediapipe. A quickstart script for Mediapipe hands is present in the Example section. 

## Example

The example code below is the example for Media-pipe hands pose estimation. Ensure that you have OpenCV installed. If not you can use the terminal command below to install OpenCV. 
```
pip3 install opencv-python
```

The code below shows the quick start example for Media-pipe hands. Appropriate comments have been added to the code which can be referred to understand the code. 
```
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
```

## Summary
In Summary, Mediapipe is an amazing tool for running ML algorithms online. For common applications like human pose detection, hands pose estimation, etc this package eliminates the need to go over the tedious process of data collection, data labeling and training a deep learning model. However, the downside is that if object detection is needed for custom objects, users still need to go through the process of labeling and training a deep learning model. Nevertheless, using the APIs in the projects, users can focus more on using the output to create impactful applications.  


## See Also:
- Gesture Control of your FireTV with Python [here](https://medium.com/analytics-vidhya/gesture-control-of-your-firetv-with-python-7d3d6c9a503b).
- MediaPipe Object Detection & Box Tracking [here](https://medium.com/analytics-vidhya/mediapipe-object-detection-and-box-tracking-82926abc50c2)
- Deep Learning based Human Pose Estimation using OpenCV and MediaPipe [here](https://medium.com/nerd-for-tech/deep-learning-based-human-pose-estimation-using-opencv-and-mediapipe-d0be7a834076)

## References
- Mediapipe Documentation. [Online]. Available: https://google.github.io/mediapipe/.
- Getting Started Documentation. [Online]. Available: https://google.github.io/mediapipe/getting_started/getting_started.html
- Mediapipe Hands Architecture. [Online]. Available: https://arxiv.org/abs/2006.10214
- MediaPipe: A Framework for Building Perception Pipelines [Online]. Available: https://arxiv.org/abs/1906.08172

/wiki/machine-learning/nlp_for_robotics/
---
date: 2022-02-05
title: NLP for robotics
published: true
---
NLP is a field of linguistics and machine learning focused on understanding everything related to human language. The aim of NLP tasks is not only to understand single words individually, but to be able to understand the context of those words.

NLP can help in robotics by enabling robots to understand and respond to natural language commands. This could be used to give robots instructions, ask them questions, and provide feedback. NLP could also be used to enable robots to interpret and respond to human emotions, allowing them to interact more naturally with people. Additionally, NLP can be used to enable robots to learn from their environment and experiences, allowing them to become more autonomous and intelligent. _(BTW, this paragraph is written by a generative transformer)_

## Transformers

Transformers are a type of neural network architecture used in natural language processing (NLP). They are based on the concept of self-attention, which allows the network to focus on specific parts of the input sentence while ignoring irrelevant words. Transformers are used for a variety of tasks, such as language modeling, machine translation, text summarization, and question answering.

The architecture is based on the idea of self-attention, which allows the network to focus on specific parts of the input sentence while ignoring irrelevant words. This allows the network to better understand the context of the sentence and make more accurate predictions.

The architecture of a transformer consists of an encoder and a decoder. The encoder takes in a sequence of words and produces a set of vectors that represent the meaning of the words. The decoder then takes these vectors and produces a prediction.

Transformers have become increasingly popular in NLP due to their ability to capture long-term dependencies in text. They have been used to achieve state-of-the-art results in a variety of tasks, such as language modeling, machine translation, text summarization, and question answering.

Transformers are a powerful tool for NLP and have revolutionized the field. They have enabled researchers to create models that can accurately capture the meaning of text and make accurate predictions.

![Transformer Architecture simplified](./assets/NLP_image1.png)

![Transformer Architecture](./assets/NLP_image2.png)
 

Encoder (left): The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.

Decoder (right): The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

Each of these parts can be used independently, depending on the task:

- Encoder-only models: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
- Decoder-only models: Good for generative tasks such as text generation.
- Encoder-decoder models or sequence-to-sequence models: Good for generative tasks that require an input, such as translation or summarization.

# Using Transformers

In general, transformers are very large. With millions to tens of billions of parameters, training and deploying these models is a complicated undertaking. Furthermore, with new models being released on a near-daily basis and each having its own implementation, trying them all out is no easy task. An easy way to implement and fine-tune transformers is the HuggingFace library.

## HuggingFace

HuggingFace is an open-source library for Natural Language Processing (NLP) that provides state-of-the-art pre-trained models for a variety of tasks such as text classification, question answering, text generation, and more. It is built on top of the popular PyTorch and TensorFlow frameworks and is designed to be easy to use and extend.

The library’s main features are:

Ease of use: Downloading, loading, and using a state-of-the-art NLP model for inference can be done in just two lines of code.

Flexibility: At their core, all models are simple PyTorch nn.Module or TensorFlow tf.keras.Model classes and can be handled like any other models in their respective machine learning (ML) frameworks.

Simplicity: Hardly any abstractions are made across the library. The “All in one file” is a core concept: a model’s forward pass is entirely defined in a single file, so that the code itself is understandable and hackable.

## Dependencies and Installation

You will need either Tensorflow or PyTorch installed.

Installing Tensorflow Guide: [Install TensorFlow 2](https://www.tensorflow.org/install/)

Installing PyTorch Guide: [Start Locally | PyTorch](https://pytorch.org/get-started/locally/)

You will also need HuggingFace’s Transformer library which can be installed through the below command

```bash
pip install transformers
```

## Simple example 1: Text classification

With the Transformer library, you can implement an NLP classification pipeline with just 3 lines of code.

```python
# Importing the libraries
from transformers import pipeline

# Load the model
classifier = pipeline('sentiment-analysis')

# Inference
result = classifier('MRSD is an awesome course!')

print(result)
```
```
Output: 'label': 'POSITIVE', 'score': 0.9998725652694702_
```
##

## Simple example 2: Natural Language Generation

```python
# Importing the libraries
from transformers import pipeline

# Init generator
generator = pipeline(task="text-generation")

# Run inference
results = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

for idx, result in enumerate(results): print(str(idx) + "| " + result['generated_text'] + '\n')
```

```
OUTPUT:

0| Hello, I'm a language model, so I'm able to write with much more precision code than I would normally, but if I want to have

1| Hello, I'm a language model, which means that anything that has a language model comes after you, but only when you really want, and never

2| Hello, I'm a language model, you can't do that here. I wanted a language that is a perfect metaphor for what we live in.

3| Hello, I'm a language model, but I could come back and say something like, we can put a value that says if a function is given

4| Hello, I'm a language model, and these are some more examples. Here's a picture of what's wrong with me.

```

##

## Simple example 3: Multimodal representation

```python
# Importing the libraries
from transformers import pipeline

# VISUAL QUESTION ANSWERING PIPELINE
vqa = pipeline(task="vqa")

# INPUTS
image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
question = "Where is the cat?"

# PREDICTION
preds = vqa(image=image, question=question)
preds = [{"score": round(pred["score"], 4), "answer": pred["answer"]} for pred in preds]
```

![Input Image](./assets/NLP_image3.png)



```
OUTPUT: [

 {'score': 0.911, 'answer': 'snow'},

 {'score': 0.8786, 'answer': 'in snow'},

 {'score': 0.6714, 'answer': 'outside'},

 {'score': 0.0293, 'answer': 'on ground'},

 {'score': 0.0272, 'answer': 'ground'}

]
```

 

## Selecting the correct model

Below is a table describing some of the basic pipeline identifiers and their use.

![List of different pipelines](./assets/NLP_image4.png)


# Fine-tuning a pretrained model

HuggingFace has a great colab notebook on fine-tuning a pre-trained model. The link to the notebook is below:

[https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/tensorflow/training.ipynb](https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/tensorflow/training.ipynb)

 
# Bias and limitations

If your intent is to use a pretrained model or a fine-tuned version in production, please be aware that, while these models are powerful tools, they come with limitations. The biggest of these is that, to enable pretraining on large amounts of data, researchers often scrape all the content they can find, taking the best as well as the worst of what is available on the internet.

To give a quick illustration, let’s go back the example of a `fill-mask` pipeline with the BERT model:

```python
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])
```
```
OUTPUT
['lawyer', 'carpenter', 'doctor', 'waiter', 'mechanic']
['nurse', 'waitress', 'teacher', 'maid', 'prostitute']
```

When asked to fill in the missing word in these two sentences, the model gives only one gender-free answer (waiter/waitress). The others are work occupations usually associated with one specific gender — and yes, prostitute ended up in the top 5 possibilities the model associates with “woman” and “work.” This happens even though BERT is one of the rare Transformer models not built by scraping data from all over the internet, but rather using apparently neutral data (it’s trained on the English Wikipedia and BookCorpus datasets).

When you use these tools, you therefore need to keep in the back of your mind that the original model you are using could very easily generate sexist, racist, or homophobic content. Fine-tuning the model on your data won’t make this intrinsic bias disappear.


/wiki/machine-learning/python-libraries-for-reinforcement-learning/
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

/wiki/machine-learning/ros-yolo-gpu/
---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2021-04-06 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: YOLO Integration with ROS and Running with CUDA GPU
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

Integrating You Only Look Once (YOLO), a real time object detection algorithm commonly used in the localization task, with ROS might pose a real integration challenge. There are many steps that are not well documented when installing the package in ROS. There is even more difficulty if one tries to switch from using the default CPU computation to using CUDA accelerated GPU computation as a ROS package.

This article serves as a step-by-step tutorial of how to integrate YOLO in ROS and enabling GPU acceleration to ensure real-time performance. The tutorial will detail two main aspects of the installation: integration with ROS and setting up CUDA. The CUDA acceleration section is stand-alone, and if you have already installed YOLO and want GPU acceleration, you can simply skip the first part.

---
## Integrating YOLO with ROS
![YOLO Demo](assets/yolo_demo.png)

To install YOLO in ROS, we will use a YOLO ROS wrapper GitHub repository [darknet_ros](https://github.com/leggedrobotics/darknet_ros). You can simply follow their instructions in the README or follow the instructions below. 

Before you start the integration, make sure you have prepared your pre-trained YOLO model weights and configurations. Based on the detection task, the pre-trained model weights may differ. If your task requires objects that are not included in the default YOLO dataset (which uses [VOC](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) or [COCO](https://cocodataset.org/#home) dataset to train), you will need to search for other pre-trained open-source projects and download their model weights and configurations to your local machine. Otherwise, you would need to train YOLO from scratch with your own dataset. The details will not be included in this article, but you may find this article helpful in learning how to do so: [Tutorial](https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/)

### Requirements

- Ubuntu: 18.04

- ROS: Melodic

- YOLO: The official YOLO ROS wrapper GitHub repo [darknet_ros](https://github.com/leggedrobotics/darknet_ros) currently only supports YOLOv3 and below. If you are using YOLOv4, try this repo instead [yolo_v4](https://github.com/tom13133/darknet_ros/tree/yolov4)

### Steps
1. #### Download the repo:
   
    ```cd catkin_workspace/src```

    ```git clone --recursive git@github.com:leggedrobotics/darknet_ros.git```

    **Note: make sure you have `--recursive` tag when downloading the darknet package**

    ```cd ../```

2. #### Build:

    ```catkin_make -DCMAKE_BUILD_TYPE=Release```

3. #### Using your own model:
   
    Within `/darknet_ros/yolo_network_config`:

      1. Add .cfg and .weights (YOLO detection model weights and configs) into /cfg and /weights folder
      
      2. Within /cfg, run `dos2unix your_model.cfg` (convert it to Unix format if you have problem with Windows to Unix format transformation)

    Within `/darknet_ros/config`:

      1. Modify "ros.yaml" with the correct camera topic
      
      2. Create "your_model.yaml" to configure the model files and detected classes

    Within `/darknet_ros/launch`:

      1. Modify "darknet_ros.launch" with the correct YAML file ("your_model.yaml")

4. #### Run:

    ```catkin_make```

    ```source devel/setup.bash```

    ```roslaunch darknet_ros darknet_ros.launch```

    After launching the ROS node, a window will automatically appear that will show the RGB stream and detected objects. You can also check the stream in [RVIZ](http://wiki.ros.org/rviz).

### ROS Topics
* #### Published ROS topics:

   * object_detector (`std_msgs::Int8`) Number of detected objects
   * bounding_boxes (`darknet_ros_msgs::BoundingBoxes`) Bounding boxes (class, x, y, w, h) details are shown in `/darknet_ros_msgs`
   * detection_image (`sensor_msgs::Image`) Image with detected bounding boxes

---
## Setting up YOLO with CUDA GPU Acceleration

You may find that running YOLO through the CPU is very slow. To increase run-time performance, you can accelerate it by using a CUDA enabled GPU. 

**Note: darknet currently only supports (last updated 2021) CUDA 10.2 with cuDNN 7.6.5 and below. If you are using CUDA 11+ or cuDNN 8.0+, you probably need to downgrade CUDA and cuDNN for darknet to work.** 

Here are the detailed instructions on installing CUDA 10.2 and cuDNN 7.6.5:

### Installing CUDA 10.2:
We will follow most of the instructions shown in this [tutorial](https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130)

**Note: If there is a `usr/local/cuda` directory in your local machine, remove it (`sudo rm -rf /usr/local/cuda`) before proceeding with the following steps below.** \
*Also, the first step will remove your display driver. This is ok, as when CUDA is reinstalled your display driver will also reinstall automatically.*

1. Remove all CUDA related files already in the machine:
   
    ```
    sudo rm /etc/apt/sources.list.d/cuda*
    sudo apt remove --autoremove nvidia-cuda-toolkit
    sudo apt remove --autoremove nvidia-*
    ```
    
2. Install CUDA 10.2:
   
    ```
    sudo apt update
    sudo apt install cuda-10-2
    sudo apt install libcudnn7
    ```

3. Add CUDA into path:
   
    ```sudo vi ~/.profile```

    Add below at the end of .profile:
    ```
    # set PATH for cuda installation
    if [ -d "/usr/local/cuda/bin/" ]; then
        export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    fi
    ```

4. Check CUDA version (make sure it is 10.2):\
  ```nvcc -V```

### Installing cuDNN separately:
1. Go to this [page](https://developer.nvidia.com/rdp/cudnn-archive), you may need to register an account with NVIDIA to access that link.

2. Download all three .deb: runtime/developer/code-sample (make sure that it's the correct version: `cuDNN 7.6.5 with CUDA 10.2`)
   
3. In Terminal:
   
    Go to the package location and install the runtime library, developer library, and (optional) code samples:

    ```
    sudo dpkg -i libcudnn7_7.6.5.32–1+cuda10.2_amd64.deb
    sudo dpkg -i libcudnn7-dev_7.6.5.32–1+cuda10.2_amd64.deb
    sudo dpkg -i libcudnn7-doc_7.6.5.32–1+cuda10.2_amd64.deb
    ``` 

4. Check cuDNN version:
   
    ```/sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn```

5. Optional: 
   
    If you cannot locate cudnn.h, or the later compilation fails with `not found cudnn.h` message:

    Copy `cudnn.h` (in `/usr/include`) to (`/usr/local/cuda/include`):\
    `sudo cp /usr/include/cudnn.h /usr/local/cuda/include`

    Copy `libcudnn*` (in `/usr/lib/x86_64-linux-gnu`) to (`/usr/local/cuda/lib64`):\
     `sudo cp /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64`


## Running YOLO with GPU Acceleration:

The process listed below will work whether you are using YOLO through the darknet_ros package or as a standalone program: 

1. Modify /darnet_ros/darknet/Makefile:
   ```
   GPU = 1
   CUDNN =1
   OPENCV = 1
   ```

   Add your GPU Architecture (ARCH) value. 
   **Note: you can find your ARCH value online [here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).**

   The values specified below correspond to a NVIDIA RTX 2070:

   ```
   -gencode=arch=compute_75,code=compute_75
   ```

2. Run `make` in `/darknet_ros/darknet`

3. Modify `/darknet_ros/darknet_ros/CmakeList.txt`:
   
   ```
   -gencode=arch=compute_75,code=compute_75
   ```

4. Run `catkin_make` in the `/catkin_ws` containing the `darknet_ros` package

**GPU Acceleration is Ready!**

---
## Summary
In this tutorial, we went through the procedures for integrating YOLO with ROS by deploying a ROS wrapper. Depending on the task, the YOLO model weights and configuration files should be added into the ROS package folder. By modifying the ROS wrapper configuration and launch file, we were able to run YOLO in ROS Melodic. 

We also demonstrated how to setup CUDA and cuDNN to run YOLO in real-time. By following our step-by-step instructions, YOLO can run with realtime performance.

## See Also
- [realsense_camera](https://roboticsknowledgebase.com/wiki/sensing/realsense/)
- [ROS](https://roboticsknowledgebase.com/wiki/common-platforms/ros/ros-intro/)

## Further Reading
- [CUDA_tutorial](https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130)
- [darknet_ros](https://github.com/leggedrobotics/darknet_ros)
- [darknet_ros_3d](https://github.com/IntelligentRoboticsLabs/gb_visual_detection_3d)

## References
- https://github.com/leggedrobotics/darknet_ros
- https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130
- https://pjreddie.com/projects/pascal-voc-dataset-mirror/
- https://cocodataset.org/#home
- https://github.com/tom13133/darknet_ros/tree/yolov4


/wiki/machine-learning/train-darknet-on-custom-dataset/
---
# date the article was last updated like this:
date: 2021-04-27 # YYYY-MM-DD
# Article's title:
title: Train Darknet on Custom Dataset
---

This serves as a tutorial for how to use YOLO and Darknet to train your system to detect classes of objects from a custom dataset. We go over installing darknet dependencies, accessing the darknet repository, configuring your dataset images and labels to work with darknet, editing config files to work with your dataset, training on darknet, and strategies to improve the mAP between training sessions.

## Install Darknet Dependencies
### Step 1:
Install Ubuntu 18.04  
Make sure you have [GPU with CC >= 3.0](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)  
  

### Step 2:
[CMake >= 3.18](https://cmake.org/download/)  
Download Unix/Linux Source   

### Step 3:
[CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)  

*Option 1:* 
Make a NVIDIA account  
Select Linux -> x86_64 -> Ubuntu -> 18.04 -> deb (local)  
Follow instructions & do Post-installation Actions  

*Option 2:*
```
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin  
$ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600  
$ wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb  
$ sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb  
$ sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub  
$ sudo apt-get update  
$ sudo apt-get -y install cuda  
$ nano /home/$USER/.bashrc  
```
Add the following to the bottom of the file  
```  
export PATH="/usr/local/cuda/bin:$PATH"  
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"  
```       
Save the file    
Close and reopen terminal  
Test for success with:    
```$ nvcc --version```  

**If it fails:**  
Restart computer    
Close and reopen terminal 
```
$ sudo apt-get autoremove    
$ sudo apt-get update    
$ sudo apt-get -y install cuda  
```

### Step 4:  
**OpenCV == 3.3.1 download from OpenCV official site:** 
```
$ git clone https://github.com/opencv/opencv  
$ git checkout 3.3.1      
``` 

### Step 5:  
**Download cuDNN v8.0.5 for CUDA 10.2:**
[cuDNN v8.0.5 for CUDA 10.2](https://developer.nvidia.com/rdp/cudnn-archive)   

**Download cuDNN Library for Linux (x86_64):**
[cuDNN Library for Linux (x86_64)](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar)

**Extract it**  
```
$ tar -xzvf cudnn-10.2-linux-x64-v8.1.0.77.tgz  
```
**Copy files to CUDA Toolkit directory**     
```
$ sudo cp cuda/include/cudnn*.h /usr/local/cuda/include   
$ sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64   
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*  
```
**If it fails:**  
Download cuDNN Runtime Library for Ubuntu18.04 x86_64 (Deb)  
Download cuDNN Developer Library for Ubuntu18.04 x86_64 (Deb)  

if there is still an issue please visit the reference site.

## Setting Up a Custom Dataset for Darknet
### Step 1: Get the images
Collect the images for your dataset (either download them from open source datasets or capture images of your own). The images must be .jpg format.

Put all the images for the dataset into a folder called “images”

### Step 2: Get the labels
#### If you already have labels:

**Check to see if the labels are in the darknet format.**
If they are, put all of the labels for the images into a folder called “labels”. 

Darknet labels are accepted as:
`<object-class> <x_center> <y_center> <width> <height>` 
Where:
`<object-class>` - integer object number from 0 to (classes-1)
`<x_center> <y_center> <width> <height>` - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
`<x_center> <y_center>` - are center of rectangle (not top-left corner)

**If you have labels for the images, but they are not in the darknet format:**
*Option 1:* Use Roboflow
Roboflow is an online tool that can convert many standard label formats between one and another. The first 1,000 images are free, but it costs $0.004 per image above that. Visit [Roboflow](https://roboflow.com/formats)

*Option 2:* Write a script and convert the labels to the darknet format yourself.

Put all of the newly converted labels for the images into a folder called “labels”.

#### If you need to make labels for the images:
Create labels for all of the images using Yolo_mark [2]. The repo and instructions for use can be found [here](https://github.com/AlexeyAB/Yolo_mark). These labels will automatically be made in the darknet format. Put all of the labels for the images into a folder called “labels”.

### Step 3: Create the text files that differentiate the test, train, and validation datasets.

  - Make a text file with the names of the image files for all of the images in the train dataset separated by a new line. Call this file “train.txt”. 

  - Make a text file with the names of the image files for all of the images in the validation dataset separated by a new line. Call this file “valid.txt”. 

  - Make a text file with the names of the image files for all of the images in the test dataset separated by a new line. Call this file “test.txt”. 

## Running Darknet  
### Step 1: Get the Darknet Repo locally and set up the data folders 
**If you do not already have the darknet github repo [1]:**
```$ git clone https://github.com/AlexeyAB/darknet  ```

**If you already have the github repo:**
```$ git pull```


### Step 2: Make Darknet 
```$ cd ./darknet  ```

**Check the Makefile and make sure the following as set as such:**  
```
GPU=1  
CUDNN=1  
OPENCV=1  
```
Save any changes and close the file.  

**Compile darknet**
```$ make  ```

### Step 3: Setup the darknet/data folder
Move the “images” and” labels” folders as well as the test.txt,  train.txt, and  valid.txt into the darknet/data folder

### Step 4: Setup the cfg folder
#### Create a new cfg folder in darknet:
```
$ mkdir custom_cfg
$ cd custom_cfg
```
#### Create the file that names the classes:
```
$ touch custom.names
```
Populate it with the names of the classes in the order of the integer values assigned to them in the darknet label format separated by new lines.

For example:
```
Light switch
Door handle
Table
```
Then in the labels, a light switch bounding box would be labeled with `0` and a table labeled with `2`.

#### Create the data file that points to the correct datasets:
```
$ touch custom.data
```
In custom.data, copy the following
```
classes= <num_classes>
train  = ./data/train.txt
valid = ./data/valid.txt
names =./custom_cfg/custom.names
backup = ./backup
eval=coco
```
Where `<num_classes>` is equal to an integer value corresponding to the distinct number of classes to train on in the dataset.

#### Create the cfg files
**Copy the cfg files to the custom cfg directory:**
```
$ cp cfg/yolov4-custom.cfg custom_cfg/
$ cp cfg/yolov4-tiny-custom.cfg custom_cfg/
```
**Edit the variables in the cfg files that are directly related to the dataset.**
> This information is taken from the darknet README but listed here for your convenience.  

If you are training YOLOv4, make these changes in ```custom_cfg/yolov4-custom.cfg```.  
If you are training YOLOv4-tiny make these changes in ```custom_cfg/yolov4-tiny-custom.cfg```.  

  - change line batch to: `batch=64`
  - change line subdivisions to: `subdivisions=16`
  - change line max_batches to:  `max_batches=<num_classes*2000>`
  > this number should not be less than number of training images, so raise it if necessary for your dataset 
  - change line steps to: `steps=<80% max_batches>, <90% max_batches>`
  - set network size `width=416 height=416` or any value multiple of 32: 
  - change classes to: `classes=<num_classes>`
  - change `filters=255` to `filters=<(num_classes + 5)x3>` in the 3 `[convolutional]` before each `[yolo]` layer, keep in mind that it only has to be the last `[convolutional]` before each of the `[yolo]` layers.


### Step 5: Download the weights files
For Yolov4, download [this file](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) and put it in darknet/custom_cfg/


For Yolov4-tiny, download [this file](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29) and put it in darknet/custom_cfg/


### Step 6: Modify the config files for mAP improvement
  - Edits will be in yolov4-tiny-custom.cfg or yolov4-custom.cfg depending on if you are running YOLOv4-tiny or YOLOv4, respectively   
  - Make sure you aren't repeating a trial already tested
  - Document your training configurations and save the config file, best weights, and the mAP graph for each iteration of training
  - See the Tips & Tricks section for recommendations to improve mAP


### Step 6: Run Darknet 
**Compile darknet again after making changes**
```$ make  ```

#### Options for how to run darknet
**To run YOLOv4 on darknet in the foreground:**  
```$ ./darknet detector train custom_cfg/custom.data custom_cfg/yolov4-custom.cfg custom_cfg/yolov4.conv.137 -map```

**To run YOLOv4-tiny on darknet in the foreground:**  
```$ ./darknet detector train custom_cfg/custom.data custom_cfg/yolov4-tiny-custom.cfg custom_cfg/yolov4-tiny.conv.29 -map ``` 

**To run YOLOv4 on darknet in the background and pass output to a log:**  
```$ ./darknet detector train custom_cfg/custom.data custom_cfg/yolov4-custom.cfg custom_cfg/yolov4.conv.137 -map  >  ./logs/darknet_logs_<date/time/test>.log 2>&1 & ``` 

**To run YOLOv4-tiny on darknet in the background and pass output to a log:**  
```$ ./darknet detector train custom_cfg/custom.data custom_cfg/yolov4-tiny-custom.cfg custom_cfg/yolov4-tiny.conv.29 -map   >  ./logs/darknet_logs_<date/time/test>.log 2>&1 &  ```


#### Check jobs to show command running:  
```$ jobs  ```

#### Show log:  
```$ tail -f ./logs/darknet_logs_<date/time/test>.log  ```

**Note**: if running in the background, Ctrl+C will not terminate darknet, but closing the terminal will  

At the end of training, find the weights in the backup folder. Weights will be saved every 1,000 iterations. Choose the weights file that corresponds with the highest mAP to save.  

**Repeat Steps 5 & 6 until a desired mAP is achieved.**


## Tips and Tricks for Training
### Train with mAP Graph
```./darknet detector train data/obj.data yolo-obj.cfg yolov4.conv.137 -map```

### Change Network Image Size
Set network size width=416 height=416 or any value multiple of 32  

### Optimize Memory Allocation During Network Resizing  
Set random=1 in cfg   
This will increase precision by training Yolo for different resolutions.  

### Add Data Augmentation
[net] mixup=1 cutmix=1 mosaic=1 blur=1 in cfg  

### For Training with Small Objects
  - Set layers = 23 instead of [this](https://github.com/AlexeyAB/darknet/blob/6f718c257815a984253346bba8fb7aa756c55090/cfg/yolov4.cfg#L895)  
  - set stride=4 instead of [this](https://github.com/AlexeyAB/darknet/blob/6f718c257815a984253346bba8fb7aa756c55090/cfg/yolov4.cfg#L892)  
  - set stride=4 instead of [this](https://github.com/AlexeyAB/darknet/blob/6f718c257815a984253346bba8fb7aa756c55090/cfg/yolov4.cfg#L989)  

### For Training with Both Large and Small Objects  
Use modified models:  
  - Full-model: [5 yolo layers](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3_5l.cfg)  
  - Tiny-model: [3 yolo layers](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny_3l.cfg)  
  - YOLOv4: [3 yolo layers](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-custom.cfg)  

### Calculate Anchors for Custom Data Set
  - ./darknet detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416  
  - Set the same 9 anchors in each of 3 [yolo]-layers in your cfg-file  
  - Change indexes of anchors masks= for each [yolo]-layer, so for YOLOv4 the 1st-[yolo]-layer has anchors smaller than 30x30, 2nd smaller than 60x60, 3rd remaining  
  - Change the filters=(classes + 5)*<number of mask> before each [yolo]-layer. If many of the calculated anchors do not fit under the appropriate layers - then just try using all the default anchors.  

## Summary
We reviewed the start to finish process of using YOLO and darknet to detect objects from a custom dataset. This included going over the darknet dependencies, dataset engineering for format compatibilities, setting up and running darknet, and improving mAP across training iterations.

## See Also:
- [Integrating darknet with ROS](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io.git/wiki/common-platforms/ros/ros-yolo-gpu.md)

## Further Reading
- Learn more about YOLO and the various versions of it [here](https://towardsdatascience.com/yolo-v4-or-yolo-v5-or-pp-yolo-dad8e40f7109)

## References
[1] AlexeyAB (2021) darknet (Version e83d652). <https://github.com/AlexeyAB/darknet>.  
[2] AlexeyAB (2019) Yolo_mark (Version ea049f3). <https://github.com/AlexeyAB/Yolo_mark>.  


/wiki/machine-learning/yolov5-tensorrt/
---
date: 2021-11-28
title: YOLOv5 Training and Deployment on NVIDIA Jetson Platforms
---

Object detection with deep neural networks has been a crucial part of robot perception. Among object detection models, single-stage detectors, like YOLO, are typically more lightweight and suitable for mobile computing platforms. In the meantime, NVIDIA has developed the Jetson computing platforms for edge AI applications, supporting CUDA GPU acceleration. Compared with desktop or laptop GPUs, Jetson’s GPUs have lower computation capacity; therefore, more care should be taken on the selection of neural network models and fine-tuning on both speed and performance. 

This article uses YOLOv5 as the objector detector and a Jetson Xavier AGX as the computing platform. It will cover setting up the environment, training YOLOv5, and the deployment commands and code. Please note that unlike the deployment pipelines of previous YOLO versions, this tutorial’s deployment of YOLOv5  doesn’t rely on darknet_ros, and at runtime, the program only relies on C++, ROS, and TensorRT.

## Jetson Xavier AGX Setup

Setting up the Jetson Xavier AGX requires an Ubuntu host PC, on which you need to install the NVIDIA SDK Manager. This program allows you to install Jetpack on your Jetson, which is a bundled SDK based on Ubuntu 18.04 and contains important components like CUDA, cuDNN and TensorRT. Note that as of 2021, Jetpack does not officially support Ubuntu 20.04 on the Jetsons. Once you have the SDK Manager ready on your Ubuntu host PC, connect the Jetson to the PC with the included USB cable, and follow the onscreen instructions. 

Note that some Jetson models including the Xavier NX and Nano require the use of an SD card image to set up, as opposed to a host PC.

After setting up your Jetson, you can then install ROS. Since Jetson runs on Ubuntu 18.04, you’ll have to install ROS Melodic. Simply follow the instructions on rog.org: [ROS Melodic Installation](http://wiki.ros.org/melodic/Installation/Ubuntu)

## Training YOLOv5 or Other Object Detectors

A deep neural net is only as good as the data it’s been trained on. While there are pretrained YOLO models available for common classes like humans, if you need your model to detect specific objects you will have to collect your own training data.

For good detection performance, you will need at least 1000 training images or more. You can collect the images with any camera you would use, including your phone. However, you need to put a lot of thought into where and how you’ll take the images. What type of environment will your robot operate in? Is there a common backdrop that the robot will see the class objects in? Is there any variety among objects of the same class? What distance/angle will the robot see the objects from? The answers to these questions will heavily inform your ideal dataset. If your robot will operate in a variety of indoor/outdoor environments with different lighting conditions, your dataset should have similar varieties (and in general, more variety = more data needed!). If your robot will only ever operate in a single room and recognize only two objects, then it’s not a bad idea to take images of only these two objects only in that room. Your model will overfit to that room in that specific condition, but it will be a good model if you’re 100% sure it’s the only use case you ever need. If you want to reinforce your robot’s ability to recognize objects that are partially occluded, you should include images where only a portion of the object is visible. Finally, make sure your images are square (1:1 aspect ratio, should be an option in your camera app). Neural nets like square images, and you won’t have to crop, pad or resize them which could undermine the quality of your data.

Once you have your raw data ready, it’s time to process them. This includes labeling the classes and bounding box locations as well as augmenting the images so that the model trained on these images are more robust. There are many tools for image processing, one of which is Roboflow which allows you to label and augment images for free (with limitations, obviously). The labeling process will be long and tedious, so put on some music or podcasts or have your friends or teammates join you and buy them lunch later. For augmentation, common tricks include randomized small rotations, crops, brightness/saturation adjustments, and cutouts. Be sure to resize the images to a canonical size like 416x416 (commonly used for YOLO). If you feel that your Jetson can handle a larger image size, try something like 640x640, or other numbers that are divisible by 32. Another potentially useful trick is to generate the dataset with the same augmentations twice; you will get two versions of the same dataset, but due to the randomized augmentations the images will be different. Like many deep learning applications, finding the right augmentations involves some trial-and-error, so don’t be afraid to experiment!

Once you have the dataset ready, time to train! Roboflow has provided tutorials in the form of Jupyter Notebooks, which contains all the repos you need to clone, all the dependencies you need to install, and all the commands you need to run:
- [The old version of the training jupyter notebook](https://colab.research.google.com/drive/1gDZ2xcTOgR39tGGs-EZ6i3RTs16wmzZQ) (The version that the authors tested on)
- [The new version of the training jupyter notebook](https://github.com/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb) (A newer version published by Roboflow)


After training is finished, the training script will save the model weights to a .pt file, which you can then transform to a TensorRT engine.

## Transforming a Pytorch Model to a TensorRT Engine

YOLOv5's official repository provides an exporting script, and to simplify the post-processing steps, please checkout a newer commit, eg. “070af88108e5675358fd783aae9d91e927717322”. At the root folder of the repository, run `python export.py --weights WEIGHT_PATH/WEIGHT_FILE_NAME.pt --img IMAGE_LENGTH --batch-size 1 --device cpu --include onnx --simplify --opset 11`. There’ll be a .onnx file generated next to the .pt file, and [netron](https://netron.app/) provides a tool to easily visualize and verify the onnx file. For example, if the image size is 416x416, the model is YOLOv5s and the class number is 2, you should see the following input and output structures:

![Figure 1. YOLOv5 onnx visualization (the input part)](../assets/yolov5_onnx_input.png)

![Figure 2. YOLOv5 onnx visualization (the output part)](../assets/yolov5_onnx_output.png)

After moving the .onnx file to your Jetson, run `trtexec --onnx=ONNX_FILE.onnx --workspace=4096 --saveEngine=ENGINE_NAME.engine --verbose` to obtain the final TensorRT engine file. The 4096 is the upper bound of the memory usage and should be adapted according to the platform. Besides, if there’s no trtexec command while TensorRT was installed, add `export PATH=$PATH:/usr/src/tensorrt/bin` to your `~/.bashrc` or `~/.zshrc`, depending on your default shell.

You don’t need to specify the size of the model here, because the input and output sizes have been embedded into the onnx file. For advanced usage, you may specify the dynamic axes on your onnx file, and at the trtexec step, you should add `--explicitBatch --minShapes=input:BATCH_SIZExCHANNELxHEIGHTxWIDTH --optShapes=input:BATCH_SIZExCHANNELxHEIGHTxWIDTH --maxShapes=input:BATCH_SIZExCHANNELxHEIGHTxWIDTH `. 

To achieve better inference speed, you can also add the `--fp16` flag. This could theoretically reduce 50% of the computation time.

## Integrating TensorRT Engines into ROS

First, under a ROS package, add the following content to your CMakeLists.txt
```
find_package(CUDA REQUIRED)
message("-- CUDA version: ${CUDA_VERSION}")

find_package(OpenCV REQUIRED)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/aarch64-linux-gnu/)
link_directories(/usr/lib/aarch64-linux-gnu/)
```

For the executable or library running the TensorRT engine, in `target_include_directories`, add `\${OpenCV_INCLUDE_DIRS} \${CUDA_INCLUDE_DIRS}`, and in `target_link_libraries`, add `\${OpenCV_LIBS} nvinfer cudart`.

In the target c++ file,  create the following global variables. The first five variables are from TensorRT or CUDA, and the other variables are for data input and output. The `sample::Logger` is defined in `logging.h`, and you can download that file from TensorRT’s Github repository in the correct branch. For example, this is the [link](https://github.com/NVIDIA/TensorRT/blob/release/8.0/samples/common/logging.h) to that file for TensorRT v8. 
```
sample::Logger gLogger;
nvinfer1::ICudaEngine *engine;
nvinfer1::IExecutionContext *context;
nvinfer1::IRuntime *runtime;
cudaStream_t stream;
```

Beyond that, you should define the GPU buffers and CPU variables to handle the input to and output from the GPU. The buffer size is 5 here because there are one input layer and four output layers in the onnx file. INPUT_SIZE and OUTPUT_SIZE are equal to their batch_size * channel_num * image_width * image_height.
```
constexpr int BUFFER_SIZE = 5;
// buffers for input and output data
std::vector<void *> buffers(BUFFER_SIZE); 
std::shared_ptr<float[]> input_data  
  = std::shared_ptr<float[]>(new float[INPUT_SIZE]);
std::shared_ptr<float[]> output_data
  = std::shared_ptr<float[]>(new float[OUTPUT_SIZE]);
```

Before actually running the engine, you need to initialize the “engine” variable, and here’s an example function. This function will print the dimensions of all input and output layers if the engine is successfully loaded.
```
void prepareEngine() {
  const std::string engine_path = “YOUR_ENGINE_PATH/YOUR_ENGINE_FILE_NAME";
  ROS_INFO("engine_path = %s", engine_path.c_str());
  std::ifstream engine_file(engine_path, std::ios::binary);

  if (!engine_file.good()) {
	ROS_ERROR("no such engine file: %s", engine_path.c_str());
	return;
  }

  char *trt_model_stream = nullptr;
  size_t trt_stream_size = 0;
  engine_file.seekg(0, engine_file.end);
  trt_stream_size = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);
  trt_model_stream = new char[trt_stream_size];
  assert(trt_model_stream);
  engine_file.read(trt_model_stream, trt_stream_size);
  engine_file.close();

  runtime = nvinfer1::createInferRuntime(gLogger);
  assert(runtime != nullptr);
  engine = runtime->deserializeCudaEngine(trt_model_stream, trt_stream_size);
  assert(engine != nullptr);
  context = engine->createExecutionContext();
  assert(context != nullptr);
  if (engine->getNbBindings() != BUFFER_SIZE) {
	ROS_ERROR("engine->getNbBindings() == %d, but should be %d",
          	engine->getNbBindings(), BUFFER_SIZE);
  }

  // get sizes of input and output and allocate memory required for input data
  // and for output data
  std::vector<nvinfer1::Dims> input_dims;
  std::vector<nvinfer1::Dims> output_dims;
  for (size_t i = 0; i < engine->getNbBindings(); ++i) {
	const size_t binding_size =
    	getSizeByDim(engine->getBindingDimensions(i)) * sizeof(float);
	if (binding_size == 0) {
  	ROS_ERROR("binding_size == 0");

  	delete[] trt_model_stream;
  	return;
	}

	cudaMalloc(&buffers[i], binding_size);
	if (engine->bindingIsInput(i)) {
  	input_dims.emplace_back(engine->getBindingDimensions(i));
  	ROS_INFO("Input layer, size = %lu", binding_size);
	} else {
  	output_dims.emplace_back(engine->getBindingDimensions(i));
  	ROS_INFO("Output layer, size = %lu", binding_size);
	}
  }

  CUDA_CHECK(cudaStreamCreate(&stream));

  delete[] trt_model_stream;

  ROS_INFO("Engine preparation finished");
}
```

At runtime, you should first copy the image data into the input_data variable. The following code snippet shows a straightforward solution. Please note that OpenCV's cv::Mat has HWC (height - width - channel) layout, while TensorRT by default takes BCHW (Batch - channel - height - width) layout data. 
```
  int i = 0;
  for (int row = 0; row < image_size; ++row) {
	for (int col = 0; col < image_size; ++col) {
  	input_data.get()[i] = image.at<cv::Vec3f>(row, col)[0];
  	input_data.get()[i + image_size * image_size] =
      	image.at<cv::Vec3f>(row, col)[1];
  	input_data.get()[i + 2 * image_size * image_size] =
      	image.at<cv::Vec3f>(row, col)[2];
  	++i;
	}
  }
```
Then, you can use the following lines of code to run the engine through its context object. The first line copies the input image data into buffers[0]. The second last line copies buffers[4], the combined bounding box output layer, to our output_data variable. In our onnx file example, this layer corresponds to 1 batch_size, 10647 bounding box entries, and 7 parameters describing each bounding box. The 7 parameters are respectively tx, ty, tw, th, object confidence value, and the two scores for the two classes, where tx, ty, tw and th means the center x, center y, width, and height of the bounding box.
```

  CUDA_CHECK(cudaMemcpyAsync(buffers[0], input_data.get(),
                         	input_size * sizeof(float), cudaMemcpyHostToDevice,
                         	stream));
  context->enqueue(1, buffers.data(), stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output_data.get(), buffers[4],
                         	output_size * sizeof(float),
                         	cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
```

The last step is the non-maximum suppression for those bounding boxes. Separately store bounding boxes according to their class id, and only keep boxes with object confidence values higher than a predefined threshold, eg. 0.5. Sort the bounding boxes from higher confidence value to lower ones, and for each bounding box, remove others with lower confidence values and intersection over union (IOU) higher than another predefined threshold, eg. 40%. To understand better about this process, please refer to [non-maximum explanation](https://www.youtube.com/watch?v=YDkjWEN8jNA).

The code of the above deployment process is available on [Github](https://github.com/Cola-Robotics/cola-object-detection). 

## Further Reading
- [TensorRT tutorial from learnopencv](https://learnopencv.com/how-to-run-inference-using-tensorrt-c-api/)
- [A thrid party implementation of YOLOv5 with TensorRT](https://github.com/wang-xinyu/tensorrtx/blob/master/yolov5)