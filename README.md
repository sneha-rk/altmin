# README #

This README would normally document whatever steps are necessary to get your application up and running.


### To run experiments and control: ###

*change options* - Don't have commandline options yet

th control.lua  or stacking-control.lua
th minibatch-stacked-alternations.lua or stacking.lua

Change options:

dataset = 'cifar' // or 'mnist'
activation = 'relu' // or 'cifar'

opt - can change
	n_epochs = 200,
	epochs = 1,
	batch_size = 500,

	inputSize = 32*32,
	outputSize = 500,
	print_every = 10,  
	learningRate = 1e-3,
	
	sizes = {}, --Sizes of inputs in various layers. Note that CIFAR haas 3*32*32
	k = 1, --Number of hidden layers 


### What is this repository for? ###

* Testing out a new way of Training Autoencoders by Alternating Minimization.

### Requirements###

* Torch7 Machine Learning Framework.
* The packages optim, image, luaposix
* wget

### How To run ###
	th train-autoencoder.lua

### Contents ###
1. mnist.t7 : Contains MNIST Dataset. Do not touch!
2. minibatch-stacked-alternations.lua - The code for training the autoencoder.
3. control.lua - the control experiment - A standard torch autoencoder
4. dataset-mnist.lua - Code to download and load the MNIST dataset.
5. dataset-cifar.lua - Code to download and load the CIFAR-10 dataset.
6. stacking.lua - Trains a multilayer autoencoder by stacking using DANTE.
7. stacking-control.lua - Trains a multilayer autoencoder by stacking using SGD.

### Who do I talk to? ###

* Purushottam Kar
* Surya Teja Chavali 
* Sneha Reddy Kudugunta
* Adepu Ravi Sankar
* Vineeth Balasubramanian