
# Final Project


## About the Project

The goal of this project was to create my own Machine Learning Model to correctly classify different logos with greater than 90% accuracy. The data was to be split in 1) Train and 2) Test data. Among the Train data, this was to be split into a 1) Train and 2) Validation dataset. The ML model that I decided to stick with is using Transfer Learning using ResNet-50, a model for classification that Tensorflow natively provides, but I pruned off the top layer and adding my own layers in.

The model is trained using the Train and Validation data. Once all the weights of the Neural Network have been set, and the model has been trained, the model is exported to a .h5 file, and this .h5 file is imported by the Test set to test the model on.

The MAIN way in which I trained the model, and the only way I ensured it didn't overfit is by AUGMENTING the data (see the train.py file)- using our original dataset, I changed images (rotated, flipped, changed brightness, contrast, etc.), to give a more variety of images, and to more emulate real world images, and did this multiple times, making our train dataset 4X as large. The ML model I used got a way higher accuracy and way closer train/validation accuracy, showing it was no longer overfitting and that it could generalize well to new, unseen data.

## Code Implementation & Technical Report

The final deliverables include a 4-page IEEE-format report, code implementation and a detailed GitHub readme file.

## Training Data

The training data set is the same for every team in this course.

You can download the training data from the Canvas page:

* ["data_train.npy"]
* ["labels_train.npy"]


## Getting Started

### Dependencies

1. Create a new conda environment `the-regularizer` with the following dependencies.

`conda create -n the-regularizer`

`conda install anaconda`

`conda install tensorflow`

To open Jupyter Notebook and see the files, in the conda terminal, while the environment is activated, do:
`jupyter notebook`

### Installation

1. Clone the repo. In the regular terminal, type:

git clone 'https://github.com/Zain3/Brand-Logo-Classification.git'

2. Setup and activate environment. In the conda terminal, type:

`conda activate the-regularizer`

3. Next, cd to whichever directory you cloned the repo, **within the conda terminal**.

## Usage

### Training
1. **Download** and put the `data_train.npy` and `labels_train.npy` in the base directory. The shapes should be (270000, n_samples) and (n_samples, ) respectively. 
2. Open the `Train.py` file, and **make sure at the bottom of the file, to type in the proper name of the dataset and training labels.**
3. In the conda terminal, do `python Train.py` . This outputs the model in a ".h5" file into the directory.


### Testing
1. Put `data_test.npy` and `labels_test.npy` in the base directory. The shapes should be (270000, n_samples) and (n_samples, ) respetively. 
2. do `python Test.py`. This reads the ".h5" model outputted by the Train.py function, and uses it on the new dataset.
3. **Make sure at the bottom of the file, to type in the proper name of the dataset and labels used for the Test dataset.**

The Test.py file will print the Loss and Accuracy, and print out a Full classification report in the terminal.

If you trained and saved a model with different name then use: `python Test.py --path <path to your model>`


## Authors
  Zain Nasrullah [email](z.nasrullah@ufl.edu)

