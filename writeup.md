# Traffic Sign Classification with LeNet

by James Medel, November 6, 2019

## Introduction

One of the first lessons human drivers learn before driving cars on the street is recognizing the traffic signs. Artificial intelligent drivers (self-driving cars) must also learn to classify these traffic signs before driving on the street. Traffic signs display valuable information to drivers that direct, inform and control their behavior in effort to make the roads more safe. 

This project involved working on a self-driving car perception problem known as traffic signalization detection. Specifically, I focused on traffic sign recognition and developed a deep learning application to classify different german traffic signs. The highlights of this solution include exploratory data analysis (EDA), data preprocessing, data augmentation, convolutional neural network (CNN) architecture interpretation from a research paper, LeNet architecture implementation using TensorFlow 1.4, building the training pipeline to train the LeNet model, developing the evaluation pipeline to evaluate the model's prediction accuracy, visualizing the model's top 5 softmax probabilities to show the certainty of the model's predictions and visualizing the inner layers of the model's network to understand what features the model focuses on. 

- Jupyter Notebook is available at the Github link: [Traffic_Sign_Classifier_with_LeNet.ipynb](https://github.com/james94/P3-Traffic-Sign-Classifier-CarND/blob/master/Traffic_Sign_Classifier_with_LeNet.ipynb)

- HTML version is available: [Traffic_Sign_Classifier_with_LeNet.ipynb](https://github.com/james94/P3-Traffic-Sign-Classifier-CarND/blob/master/Traffic_Sign_Classifier_with_LeNet.html)


## Dataset Exploration

During the dataset exploration task, I used exploratory data analysis (EDA) to analyze, summarize and visualize the contents of the dataset. Exploring the data before doing anything with it is important because if you run into a problem, you will have an idea where to look for potential sources of error.

### Dataset Summary

`Submission includes basic summary of the data set`

The lifelike traffic sign database we were required to use for the deep learning application was the [German Traffic Sign Recognition Benchmarks (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) dataset. GTSRB has the following properties: more than 40 classes, more than 50,000 images in total and lifelike database. The dataset was provided as pickle files for the training, validation and testing set. The pickled data has 4 key/value pairs and the keys are 'features', 'labels', 'sizes' and 'coords'. The 'features' key value is a 4D array containing raw pixel data of the traffic sign names (num of examples, width, height, channel). The 'labels' key value is a 1D array containing the class id of the traffic sign. Later this class id is used in the [signnames.csv]() to look up the traffic sign text names for exploratory visualization. The 'sizes' key value is a list containing tuples (width, height) representing the original width and height of the image. The 'coords' key value is a list containing tuples (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. 

For the project, I loaded the 'features' and 'labels' pickled data for the training, validation and testing set. I calculated the summary of the data set using numpy. I passed the data from the summary into a pandas dataframe to visualize it in a table:

| Summary | Value |
| -------- | ----- |
| Total Training Examples | 34,799 |
| Total Validation Examples | 4,410 |
| Total Testing Examples | 12,630 |
| Image Shape | (32, 32, 3) |
| Total Classes | 43 |
| Total Training Set Count of Class ID 0 | |
| Total Training Set Count of Class ID 1 | 1980 |

### Exploratory Visualization

`Submission includes an exploratory visualization on the dataset`

## Design and Test a Model Architecture

### Preprocessing

`Submission describes the preprocessing techniques used and why these techniques were chosen`

### Model Architecture

`Submission provides details of the characteristics and qualities of the architecture`

includes the type of model used, the number of layers, the size of each layer.

Visualizations emphasizing the particular qualities of the architecture are encouraged.

### Model Training

`Submission describes how the model was trained`

Discuss what optimizer was used, batch size, number of epochs and values for hyperparameters.

### Solution Approach

`Submission describes the approach to finding a solution`

Accuracy on the validation set is 0.93 or greater

## Test a Model on New Images

### Acquiring New Images

`Submission includes five new German Traffic signs found on the web and visualized`

Discuss the particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify

### Performance on New Images

`Submission documents the performance of the model when tested on captured images`

The performance on the new images is compared to the accuracy results of the test set

### Model Certainty - Softmax Probabilities

`Submission discusses how certain or uncertain the model is of its predictions`

Top five softmax probabilities of the predictions on captured images are outputted

## Suggestitons to Make Your Project Stand Out!

### Augment the Training Data

### Analyze New Image Performance in More Detail

### Create Visualizations of the Softmax Probabilities

### Visualize Layers of the Neural Network
