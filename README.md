## Project: Build a Traffic Sign Recognition Program

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Top 5 Softmax Probabilities of new images:

![top5_softmax_probabilities.jpg](assets/data/images/output/top5_softmax_probabilities.jpg)

Overview
---

The purpose of this project was to build a traffic sign recognition application using knowledge acquired from deep neural networks (DNN) and convolutional neural networks (CNN). A CNN model was built, trained and validated, so it can classify traffic sign images using Tenorflow and the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Finally, the model was tested on German Traffic signs from the internet.

Contents
---

- [Traffic_Sign_Classifier_with_LeNet.ipynb](Traffic_Sign_Classifier_with_LeNet.ipynb): application code
- [writeup.md](writeup.md): explains my traffic sign recognition application, shortcomings, potential improvements and each rubric point with a description of how that point was addressed.
- **README.md**: provides overview of the project and how to set it up
- **assets/**: folder contains image data results captured after running the application, lenet modified architecture, previous notebook checkpoints, writeup help resources and csv for mapping sign name ID to sign name text.
- lenet_model.model.data-00000-of-00001, lenet_model.index, lenet_model.meta: the saved CNN model

Results after Running Application Code:
---

Jupyter Notebook Result

To see the output result after the code was executed, visit the jupyter notebook: [Traffic_Sign_Classifier_with_LeNet.ipynb](Traffic_Sign_Classifier_with_LeNet.ipynb)

HTML Version of Jupyter Notebook Result:

To see the output result after the Jupyter notebook was dowloaded as an HTML file, visit the HTML file:
[Traffic_Sign_Classifier_with_LeNet.html](Traffic_Sign_Classifier_with_LeNet.html)

The Project
---

The goals / steps of this project were the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies

This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set at Kaggle link [Traffic Signs Pickled Dataset](https://www.kaggle.com/tomerel/traffic-signs-pickled-dataset). This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Jupypter notebook and the writeup template.

```sh
git clone https://github.com/james94/P3-Traffic-Sign-Classifier-CarND

# OpenCV, Tensorflow 1.4 may need to be installed

cd P3-Traffic-Sign-Classifier-CarND
jupyter notebook Traffic_Sign_Classifier_with_LeNet.ipynb
```


