# README #

### To run experiments and control: ###

th stacking-control.lua \\
th stacking.lua

All previous implementations may be found in the previous folder
*change options* - Don't have commandline options yet

### Change options

dataset = 'cifar' // or 'mnist' \\
activation = 'relu' // or 'sigmoid'

opt = 
{
	epochs = 20,
	n_finetune = 1,			  -- Number of finetuning layers

	batch_size = 500,
	print_every = 10,  
	train_size = 60000,
	test_size = 10000,
	sizes = {1024, 500, 250}, -- Sizes of inputs in various layers.
	channels = 1,			  -- CIFAR 10 has 3 channels.

	learningRate = 1.0e-3, 	  -- SET PROPERLY.
	n = 2					  -- Number of total epochs
}


### What is this repository for? ###

* Testing out a new way of Training Autoencoders by Alternating Minimization.

### Requirements###

* Torch7 Machine Learning Framework.
* The packages optim, image, luaposix
* wget

### How To run ###
	th stacking.lua or stacking-control.lua

### Contents ###
1. mnist.t7 : Contains MNIST Dataset. Do not touch!
2. dataset-mnist.lua - Code to download and load the MNIST dataset.
3. dataset-cifar.lua - Code to download and load the CIFAR-10 dataset.
4. stacking.lua - Trains a multilayer autoencoder by stacking using DANTE.
5. stacking-control.lua - Trains a multilayer autoencoder by stacking using SGD.

### Who do I talk to? ###

* Sneha Reddy Kudugunta
* Surya Teja Chavali 
* Adepu Ravi Sankar
* Vineeth Balasubramanian
* Purushottam Kar
