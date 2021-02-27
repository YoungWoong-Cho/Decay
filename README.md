# Decay: An Organic Degradation

<strong>Before reading: Please refer to following [paper](https://arxiv.org/pdf/1703.10593.pdf) for more information about CycleGAN.</strong>

<blockquote> Our project began with the question, <i>“what if inorganic things <u>‘decay’</u> like organic things?”</i> </blockquote>

***Abstract—The importance of sustainable design is increasing. Through deep convolutional neural network, we imagined the world where everything is organic, and thus perishes.<br>
A deep neural network was trained to predict the "decayed" appearance of an arbitrary input image.<br>
A dataset that consists of several hundred images of fresh/rotten apples, oranges, and bananas were used to train the neural network.
pix2pix, CycleGAN, and DiscoGAN were considered as our neural network model, and CycleGAN was found to perform the best.***

### INTRODUCTION
The dichotomy that we established between **inorganic** and **organic** materials are no longer applicable.
Witnessing the immortality of inorganic materials threatening the lives and existence of organic materials, we are creating a hypothetical situation where everything in the world decays and perishes. 

### PROCESS
Multiple training images of fresh and rotten organic food are fed into our model, with which the model will update its parameters.
Various neural network architectures were considered.
<ol>
<li> <strong>pix2pix</strong><br>
Our first consideration was <a href=https://arxiv.org/pdf/1611.07004.pdf>pix2pix</a>.

However, it was not successful, since pix2pix required a set of <strong>paired dataset</strong>.
<li> <strong>DiscoGAN</strong><br>
Our next consideration was <a href=https://arxiv.org/pdf/1703.05192.pdf>DiscoGAN</a>. It was chosen since it not only could take in <strong>unpaired dataset</strong> but also was capable of performing <strong>geometry change</strong>.<br>
Following figure shows the resulting pair of image.<br>
<img src="images/pix2pix.jpg" alt="pix2pix">

</ol>
