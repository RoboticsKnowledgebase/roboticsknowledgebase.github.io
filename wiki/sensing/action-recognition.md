Traffic Policeman Gesture Recognition With Spatial Temporal Graph Convolution Network

Introduction. With the rise of the popularity of self-driving cars, there would be more scenarios where the autonomous cars would be used in environments which are not properly designed for them. Building an automatic gesture recognition system is essential for self-driving cars, especially in the developing countries where the traffic signal infrastructure could be not present or working. In this project, we focus on the video understanding of the automatic gesture recognition system to classify the gesture of traffic policemen who stands at traffic intersections and give commands to the oncoming traffic. We use skeleton data for pose recognition because -
• Skeleton recognition systems like OpenPose work
well, acting as an excellent noise removal utility.
• All the information about gesture is present in the
skeleton.
• Low dimensional skeleton representation allows the fine tuning of feature extraction networks with very little data.

The Dataset

Owing to the lack of available public datasets for the task of policeman gesture recognition, we created our own dataset for the purpose. We chose three gestures to record, namely, left turn command, right turn command and parallel traffic command. The three gestures are shown in Figure 1. We recorded 45 videos at 30 f ps of 10 seconds duration of the three of the authors performing the three gestures. The videos were recorded in the Robotics institute at Carnegie Mellon University with an iPhone 7 Plus camera. We created a balanced train/test split of 36/9 videos for further processing. We cropped the video to the aspect ratio of 1.43 as required by the feature extraction network. 

Our Approach

We used OpenPose [1] to extract 18 keypoints from the human body. This OpenPose has been trained on Kinetic dataset [2] from DeepMind. Then we used a dynamic skeleton based approach - Spatial Temporal Graph Convolutional Networks (ST-GCN) [3] to classify the actions in the video consisting of approximately 300 frames. Firstly, we passed our 10 seconds long videos through the OpenPose and stored 18 keypoints with their confidence scores in the .json files. This data has been recorded for all the 300 frames in the video. Then we used these pose estimation results as an input to our ST-GCN network. ST-GCN network is a graph based network which follows Spatial Perspective stream where the convolution filters are applied directly on the graph nodes and their neighbors (Bruna et al. 2014; Niepert, Ahmed, and Kutzkov 2016).In ST-GCN, multiple layers of spatial-temporal graph convolution operations are applied on the input data from OpenPose which generates higher-level feature maps on the graph. Then standard Softmax classifier is used to classify the action classes. The entire model is trained in an end-to end manner with backpropagation.

Network Architecture

A batch normalization layer is applied to normalize input skeleton data. The ST-GCN model consists of 9 layers (ST-GCN units). The first three layers have 64 channels as output. The following three layers have 128 channels as output. And the last three layers have 256 channels as output. These layers have temporal kernel size of 9. The Resnet mechanism is applied on each ST-GCN unit. And we randomly dropout the features at 0.5 probability after each ST-GCN unit to avoid overfitting. The strides of the 4-th and the 7-th temporal convolution layers are set to 2 as pooling layer. After that, a global pooling was performed on the resulting tensor to get a 256 dimension feature vector for each sequence. Finally, we feed them to a Softmax classifier. The models are learned using stochastic gradient descent with a learning rate of 0.01. We decay the learning rate by 0.1 after every 10 epochs. To avoid overfitting, we perform data augmentation as well. Also global pooling handles input sequence of any length, as in the input frames doesn’t always have to be a 300 frames exact.

Results 

We used a pre-trained network and stripped-off last Softmax layer and feed in our own layer which consists of 3 categories (3 output channels) as present in our use case. Then we trained it for 50 epochs for 36 training examples and then evaluated on 9 testing examples. We achieved a perfect training set accuracy and a testing set accuracy of 88.9%. 

Conclusion
We demonstrated an effective method to train a gesture recognition system with very little data custom data. We accomplished this by using existing tools like OpenPose and finetuning pretrained network ST-GCN on the limited amount of training data. We demonstrated that it is possible to get high accuracy with pose recognition even with small amounts of data. For future work, it would be interesting to work with real world dataset and a real-time pipeline for in-ference to be useful with self-driving cars.

Bibliography

[1] Cao, Zhe, et al. "Realtime multi-person 2d pose esti- mation using part affinity fields." Proceedings of the IEEE - https://arxiv.org/pdf/1611.08050.pdf

Conference on Computer Vision and Pattern Recognition. 2017.
[2] Kay, Will, et al. "The kinetics human action video dataset." arXiv preprint arXiv:1705.06950. 2017. - https://arxiv.org/pdf/1705.06950.pdf

[3] Yan, Sijie, Yuanjun Xiong, and Dahua Lin. "Spatial tem- poral graph convolutional networks for skeleton-based action recognition." Thirty-Second AAAI Conference on Artificial Intelligence. 2018. https://arxiv.org/pdf/1801.07455.pdf

