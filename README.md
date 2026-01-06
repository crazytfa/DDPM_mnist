# DDPM_mnist
This project implements image generation based on DDPM and also demonstrates MNIST data generation guided by different classifiers.

`train_ddpm.py` is used for training the diffusion model;
`train_mnist_classifier.py` is used for training the ResNet classifier;
`wekclassifier.py` is used for training a simple CNN classifier;
`allcon.py` is used for generating images, with the option to use either a CNN or ResNet classifier for guidance;
`clipcon.py` is used for generating images guided by CLIP;
`test_clip.py` is used to test CLIP's classification capabilities on MNIST data.
The specific model architecture code for the diffusion model is located in `src\diffusion`.
The environment configuration file required for all the code is `environment.yml`.

ref：
https://github.com/juraam/stable-diffusion-from-scratch/tree/main
