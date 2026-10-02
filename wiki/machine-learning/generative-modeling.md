---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances.
# You should set the date the article was last updated like this:
date: 2024-12-01 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Generative modeling
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.

---
This blog is supposed to be a junction that connects some of the important concepts in generative modeling. It provides high-level information about generative AI and its importance, popular methods, and key equations. Readers can find more detailed information in the references provided.

## Introduction
In recent years, generative models have taken the machine learning world by storm, revolutionizing our ability to create and manipulate data across various domains. This blog post will explore the fascinating world of generative modeling, from its fundamental concepts to cutting-edge applications.
### Introduction to Generative Modeling
Generative modeling is a subfield of machine learning focused on creating new data samples that mimic the characteristics of a given dataset. Unlike discriminative models, which predict labels or outcomes (e.g., p(y|x)), generative models learn the underlying distribution p(x) or joint distribution p(x,y). This enables them to generate novel samples that resemble real data.

The goal of generative models is to approximate these complex, high-dimensional data distributions. For instance, if we represent the data distribution as p(x), a generative model attempts to learn this function such that it can generate $\bar{x} \sim p(x)$, where $\bar{x}$ is a new, generated sample. Recent advances in deep learning have significantly improved the ability of these models to generate realistic images, coherent text, and more.

#### Fundamental Methods in Generative Modeling
##### Variational Autoencoders (VAEs)
VAEs are probabilistic models that encode data into a latent space (z) and then decode it back to reconstruct the original data. The generative process assumes:

$$\begin{aligned}
p(x) &= \int p(x | z) p(z) \, dz
\end{aligned}$$

where $p(z)$ is the prior distribution (often a Gaussian), and $p(x|z)$ is the likelihood. The VAE optimizes a lower bound on the data log-likelihood, known as the Evidence Lower Bound (ELBO):

$$\begin{aligned}
\mathcal{L}_{\text{ELBO}} &= \mathbb{E}_{q_\phi(z | x)}[\log p_\theta(x | z)] - D_{\text{KL}}(q_\phi(z | x) \| p(z))
\end{aligned}$$

The first term encourages the decoder $p_\theta(x|z)$ to reconstruct $x$ from $z$, while the second term ensures that the approximate posterior $q_\phi(z|x)$ remains close to the prior $p(z)$. This probabilistic framework allows VAEs to generate new data by sampling $z \sim p(z)$ and passing it through the decoder.


##### Generative Adversarial Networks (GANs)
GANs consist of two neural networks: a generator $G(z)$ and a discriminator $D(x)$. The generator learns to map random noise $z \sim p_z$ to data $x$, while the discriminator attempts to distinguish between real data $x \sim p_{\text{data}}$ and generated data $\tilde{x} = G(z)$.

$$\begin{aligned}
\mathcal{L}_{\text{GAN}} = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))].
\end{aligned}$$
The generator aims to minimize $\mathcal{L}_{\text{GAN}}$, while the discriminator maximizes it:
$$\begin{aligned}
G^* = \arg \min_G \max_D \mathcal{L}_{\text{GAN}}.
\end{aligned}$$
This adversarial process ensures the generator produces increasingly realistic samples, making GANs highly effective for generating images and videos.

##### Flow-based Models
Flow-based models transform data $x$ into a simpler latent representation $z$ through an invertible transformation $f$. This allows the likelihood $p(x)$ to be computed exactly:
$$\begin{aligned}
z \sim \pi(z), x = f(z), z = f^{-1}(x) \\
p(x) = \pi(z) \left| \det \frac{\partial z}{\partial x} \right| = \pi(f^{-1}(x)) \left| \det \frac{\partial f^{-1}}{\partial x} \right|
% p(x) = \pi(z) \left| \det \frac{\partial f}{\partial u} \right|^{-1},
\end{aligned}$$
where $z = f^{-1}(x)$.

By designing $f$ as a series of bijective layers, flow-based models enable efficient sampling and exact density estimation. Architectures like RealNVP and Glow use this property to generate high-quality images.


##### Diffusion Models
Diffusion models generate data by learning to reverse a process that gradually adds noise to data. Starting from noise $x_T$, they iteratively denoise to produce a coherent sample $x_0$.

The forward process is defined as:
$$\begin{aligned}
q(x_t | x_{t-1}) = \mathcal{N}(x_t | \sqrt{\alpha_t} x_{t-1}, \sqrt{1 - \alpha_t} I),
\end{aligned}$$
where $\alpha_t$ controls the noise schedule. The reverse process is modeled as:
$$\begin{aligned}
p(x_{0:T}) = p(X_T)\prod_{t=1}^T p(x_{t-1} | x_t) \\
p(x_{t-1} | x_t) = \mathcal{N}(x_{t-1} | \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
\end{aligned}$$
Diffusion models optimize the following loss:
$$\begin{aligned}
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{q(x_t | x_{t-1})} \left[ ||x_t - \mu_\theta(x_{t-1}, t)||^2 \right]
\end{aligned}$$
These models have shown remarkable success in tasks like image generation (e.g., DALL-E 2 and Stable Diffusion).

##### Autoregressive Generation
Autoregressive models generate data sequentially, predicting each element conditioned on previous ones. For a sequence $x = (x_1, x_2, \dots, x_T)$, the joint probability is factorized as:
$$\begin{aligned}
p(x) = \prod_{t=1}^T p(x_t | x_1, \dots, x_{t-1}).
\end{aligned}$$
###### Examples:
**PixelCNN** for images, where each pixel is generated conditioned on previously generated pixels.
**GPT** for text, where each token is generated based on preceding tokens.
These models excel in generating coherent sequences, particularly in text and music.

#### Applications of Generative Models
Generative models have found applications across various domains:
- **Text Generation**: From language translation to chatbots, generative models are powering advanced natural language processing systems.
- **Image Generation and Manipulation**: Models like DALL-E and Midjourney can create photorealistic images from text descriptions, while others enable sophisticated image editing and style transfer.
- **Video Generation**: Recent advancements allow for the creation of short video clips from text prompts or the manipulation of existing videos.
- **Audio Generation**: From text-to-speech systems to music composition, generative models are pushing the boundaries of audio synthesis.

#### Foundation Models
Foundation models represent a paradigm shift in AI, where large-scale models trained on vast amounts of data serve as a basis for a wide range of downstream tasks. These models, such as BERT and GPT-3, have demonstrated remarkable zero-shot and few-shot learning capabilities.
In robotics, foundation models are being explored for tasks such as visual understanding, task planning, and natural language instruction following. They hold the potential to bridge the gap between perception and action in embodied AI systems.

#### Conclusion
Generative models have come a long way in recent years, pushing the boundaries of what's possible in artificial intelligence. As research continues to advance, we can expect even more impressive applications and capabilities to emerge, reshaping industries and opening new frontiers in AI-driven creativity and problem-solving.

### References
##### VAEs
- Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).
##### GANs
- Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. "Generative adversarial nets." In Advances in neural information processing systems, pp. 2672-2680. 2014.
##### Flow-based Models
- Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. "Density estimation using real nvp." arXiv preprint arXiv:1605.08803 (2016).
- Rezende, Danilo Jimenez, and Shakir Mohamed. "Variational inference with normalizing flows." International Conference on Machine Learning (ICML). 2015.
##### Diffusion Models
- Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." arXiv preprint arXiv:2006.11239 (2020).
##### Autoregressive Generation
- van den Oord, Aaron, Nal Kalchbrenner, and Koray Kavukcuoglu. "Pixel recurrent neural networks." International Conference on Machine Learning (ICML). 2016.
- Radford, Alec, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. "Improving language understanding by generative pre-training." OpenAI preprint (2018). (For GPT and autoregressive text models.)

### Related
- [Lilian Weng's blog on VAEs](https://lilianweng.github.io/posts/2018-08-12-vae/)
- [Lilian Weng's blog on Flow-based models](https://lilianweng.github.io/posts/2018-10-13-flow-models/)
- [Lilian Weng's blog on GANs](https://lilianweng.github.io/posts/2017-08-20-gan/)
- [Lilian Weng's blog on Diffusion models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
