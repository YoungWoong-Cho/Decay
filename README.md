# Decay: An Organic Degradation

<strong>Before reading: Please refer to following [paper](https://arxiv.org/pdf/1703.10593.pdf) for more information about CycleGAN.</strong>

<blockquote> Our project began with the question, <i>“what if inorganic things <u>‘decay’</u> like organic things?”</i> </blockquote><br>

### TL;JUST RUN THE CODE
<ul>
<li>opencv-python==4.2.0
<li>Pillow==7.1.2
<li>tensorflow==2.4.1
<li>torch==1.7.0
<li>torchvision==0.8.1
</ul>
Type in the following command:

    python train.py

If you want to train the model with your own model,

    python train.py --dataset your2data --epochs 100 --decay_epochs 80

If your device supports CUDA, it will be activated automatically. If you want to explicitly turn on CUDA,

    python train.py --cuda

After training, you can test your model using your custom input image. The images to be tested should be located under "./test".<br>
To specify the model, use the following command:

    python test.py --model weights/fruits2rotten/G_A2B.pth --cuda

For help, type

    python train.py --help
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
<img src="https://github.com/YoungWoong-Cho/Decay/blob/youngwoong/images/pix2pix.PNG" alt="pix2pix"><br>
However, it was not successful, since pix2pix required a set of <strong>paired dataset</strong>. It was almost impossible to prepare several hundreds of fresh-rotten pair of images.
<li> <strong>DiscoGAN</strong><br>
Our next consideration was <a href=https://arxiv.org/pdf/1703.05192.pdf>DiscoGAN</a>. It was chosen since it not only could take in <strong>unpaired dataset</strong> but also was capable of performing <strong>geometry change</strong>. Since an apple, an orange, or a banana goes through a morphological change when rotten, DiscoGAN seemed to be a good choice.<br>
Following figure shows the resulting images after 50 epochs of training with apple2rotten dataset, which consists of 184 images of fresh apple and 271 images of rotten apple.<br>
<img src="https://github.com/YoungWoong-Cho/Decay/blob/youngwoong/images/discoGAN.PNG" alt="discoGAN"><br>
0.A.jpg is the original image of an apple, 0.AB.jpg is the image that is mapped to the domain of rotten apple, and 0.ABA.jpg is the reconstructed image, which is mapped back to the original domain. As it can be seen, the translated image(0.AB.jpg) not only shows the change in colors but also in its contours.<br>
However, the transformation was not performing well at all for the images that are not apples. Take a look at following transformations.<br>
<img src="https://github.com/YoungWoong-Cho/Decay/blob/youngwoong/images/discoGAN2.PNG" alt="discoGAN"><br>
<img src="https://github.com/YoungWoong-Cho/Decay/blob/youngwoong/images/discoGAN3.PNG" alt="discoGAN"><br>
  When the model attempts to convert a non-apple image into an apple image, it tries to <i>force</i> to translate the image into an apple; thus, drastic failure.
 <li> <strong>CycleGAN</strong><br>
Our next consideration was <a href=https://arxiv.org/pdf/1703.10593.pdf>CycleGAN</a> because it could preserve the <strong>geometric characteristics</strong> of the input image.
Following figures show the resulting images after 200 epochs of training with banana2rotten dataset, which consists of 342 images of fresh banana and 484 images of rotten banana.<br>
<img src="https://github.com/YoungWoong-Cho/Decay/blob/youngwoong/images/cycleGAN_banana.png" alt="cycleGAN_banana"><br>
<img src="https://github.com/YoungWoong-Cho/Decay/blob/youngwoong/images/cycleGAN_banana2.png" alt="cycleGAN_banana2"><br>
It seems working! However, since our dataset mostly consists of banana, the model performed well only for the <i>yellow</i> images.<br>
Therefore, we decided to train the model using <i>all images of the fruits</i> - apples, oranges, and bananas altogether, so that our model can be ready for more diverse colors.<br>
Following image shows some of the resulting images generated by the model while being trained.<br>
<img src="https://github.com/YoungWoong-Cho/Decay/blob/youngwoong/images/cycleGAN_fruits3.PNG" alt="cycleGAN_fruits3"><br>
It can be seen that the model is trained to translate the images of red, yellow, orange, and green.</br>
</ol>

### RESULT
Following images are generated by our model that is trained to predict the decay of the input image.
<img src="https://github.com/YoungWoong-Cho/Decay/blob/youngwoong/images/cycleGAN_fruits.png" alt="cycleGAN_banana"><br>
<img src="https://github.com/YoungWoong-Cho/Decay/blob/youngwoong/images/cycleGAN_fruits2.png" alt="cycleGAN_banana"><br>
The model is trained with 505 images of fresh fruits, and 690 images of rotten fruits. It was trained for 200 epochs.
