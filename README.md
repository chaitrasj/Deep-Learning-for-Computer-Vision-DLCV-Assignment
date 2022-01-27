# Deep Learning for Computer Vision (DLCV) Assignment
The repository describes about the experiments conducted as a part of the assignemnt for DLCV course- [DS 265](https://val.cds.iisc.ac.in/DLCV/) at the Indian Institute of Science. For more details about each experiment and the analysis and plots, please refer to the [Project report](https://github.com/chaitrasj/Deep-Learning-for-Computer-Vision-DLCV-Assignment/blob/main/Chaitra_Jambigi_A3.pdf).

## 1. Text generation using RNN
Implemented a character-level Recurrent Neural Network that models the probability distribution of the next character in a sequence, given the previous
characters in the sequence. At each time step, the input to the network is a one-hot encoding of a single character. Dataset used is the [Harry Potter](https://github.com/chaitrasj/Deep-Learning-for-Computer-Vision-DLCV-Assignment/tree/main/RNN/Harry%20Potter) novel text data.
- Implemented the character-level RNN from scratch without using any libraries.
- Tuned for Hyperparameters like sequence length, temperature, number of hidden nodes, number of layers.
- Demo generated text at the end of training is available.
- Convergence speed by different optimisers is analysed.
- Diversity in text generation by varying Temperature parameter is analysed.

### RNN Training
1. The code is present in RNN folder as [RNN.ipynb](https://github.com/chaitrasj/Deep-Learning-for-Computer-Vision-DLCV-Assignment/blob/main/RNN/RNN.ipynb).
2. Proper comments are added in the notebook.
3. ‘Generated_text.txt’ is the generated 1000 character sequence.
4. ‘Temperature_results.txt’ are the text outputs for different temperature values.



## 2. Generative Adversarial Networks (Vanilla GAN)
Trained a Generative Adversarial Network (GAN) to generate images using MNIST dataset. Used the Deep Convolutional GAN ([DCGAN](https://arxiv.org/pdf/1511.06434.pdf)) architecture.
- Implemented the GAN using Pytorch framework.
- Tuned for hyperparameters such as number of layers, learning rate, weight initialisation, architectural changes.
- Explored the training tricks for GANs based on the [blog](https://github.com/soumith/ganhacks).
- Analysed the convergence criteria based on Generator and Discriminator loss plots using Tensorboard.
- Used the trained Generator to traverse in the latent space of GAN to generate diverse MNIST images.
- Tested the accuracy of the generated images on the classifier pre-trained on MNIST.

### GAN Training
1. All codes are present in GAN folder.
2. Code is tested on torch 1.4.0, torchvision 0.5.0
3. Apart from these it uses matplotlib, numpy, os, collections, tensorboardX packages.
4. ‘train.py’ is the main training file. The code saves models and generated images after every 10 epochs and Tensorboard plots. You can specify the folder in which you want to save it by giving the argument --name ‘Name_of_folder’. Default value of the name is ‘1_Model’. 
5. ‘Classifier.py’ file is the simple file which trains the classifier and the trained classifier model is saved as ‘classifier.pth.tar’.
6. ‘Interpolation.ipynb’ is the code which loads the trained model and generates the Latent space traversal images. As of now 918 images are generated and saved as ‘traversal_gan.png’. You can modify the name by changing --name in the ipynb.
7. ‘Generation.ipynb’ is used to generate images and test the Classifier accuracy. It also gives the distribution of generated images. It saves 10 generated images with name ‘Generated_images.png’ Also the file displays the predicted class labels for each of these 10 images.



## 3. Variational Auto-encoders
Trained a Variational Autoencoder (VAE) to generate images using MNIST dataset. Used the Deep Convolutional GAN ([DCGAN](https://arxiv.org/pdf/1511.06434.pdf)) architecture.
- Implemented the VAE using Keras framework with Tensorflow as the backend.
- Tuned for hyperparameters such as number of layers, learning rate, weight initialisation, architectural changes.
- Used the trained Generator to traverse in the latent space of GAN to generate diverse MNIST images.
- Compared the generated images using VAE verses the ones generated with the GAN.
- Tested the accuracy of the generated images on the classifier pre-trained on MNIST.

### VAE Training
1. All codes are present in VAE folder
2. Code is tested on tensorflow-gpu 2.2.0, tensorflow 2.1.0, keras 2.3.1.
3. For plotting purposes, torchvision 0.5.0 and torch 1.4.0 are used. Apart from these, numpy and os packages are used.
4. ‘train.py’ is the training file. Running this file will create a model at the end of 30 epochs with name ‘Model_1.h5’. We can change the name by giving name in --name
5. It generated the latent space traversal output with name ‘traversal_vae_1.png’ which can be changed at --name_img. 918 images are generated
