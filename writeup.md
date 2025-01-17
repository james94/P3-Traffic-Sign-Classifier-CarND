# Traffic Sign Classification with LeNet

by James Medel, November 11, 2019

## Introduction

One of the first lessons human drivers learn before driving cars on the street is recognizing the traffic signs. Artificial intelligent drivers (self-driving cars) must also learn to classify these traffic signs before driving on the street. Traffic signs display valuable information to drivers that direct, inform and control their behavior in effort to make the roads more safe. 

This project involved working on a self-driving car perception problem known as traffic signalization detection. Specifically, I focused on traffic sign recognition and developed a deep learning application to classify different German Traffic Signs using Tensorflow. The highlights of this solution include:

- Exploratory data analysis (EDA)
- Data preprocessing, data augmentation
- LeNet convolutional neural network (CNN) architecture interpretation from a research paper
- CNN architecture implementation using TensorFlow 1.4
- Building the training pipeline to train the CNN model
- Developing the evaluation pipeline to evaluate the model's prediction accuracy
- Visualizing the model's top 5 softmax probabilities to show the certainty of the model's predictions
- Visualizing the inner layers of the model's network to understand what features the model focuses on. 

For code reference, visit either the Jupyter notebook or the HTML version of the Jupyter notebook:

- Jupyter Notebook with code is available at the Github link: [Traffic_Sign_Classifier_with_LeNet.ipynb](Traffic_Sign_Classifier_with_LeNet.ipynb)

- HTML version with code is available: [Traffic_Sign_Classifier_with_LeNet.html](Traffic_Sign_Classifier_with_LeNet.html)


## Dataset Summary & Exploration

During the dataset exploration task, I used exploratory data analysis (EDA) to analyze, summarize and visualize the contents of the dataset. Exploring the data before doing anything with it is important because if you run into a problem, you will have an idea where to look for potential sources of error.

### Dataset Summary

The lifelike traffic sign database we were required to use for the deep learning application was the [German Traffic Sign Recognition Benchmarks (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) dataset. GTSRB has the following properties: more than 40 classes, more than 50,000 images in total and lifelike database. The dataset was provided as pickle files for the training, validation and testing set. The pickled data has 4 key/value pairs and the keys are 'features', 'labels', 'sizes' and 'coords'. The 'features' key value is a 4D array containing raw pixel data of the traffic sign names (num of examples, width, height, channel). The 'labels' key value is a 1D array containing the class id of the traffic sign. Later this class id is used in the [signnames.csv](assets/images/../data/images/web_signnames.csv) to look up the traffic sign text names for exploratory visualization. The 'sizes' key value is a list containing tuples (width, height) representing the original width and height of the image. The 'coords' key value is a list containing tuples (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. 

For the project, I loaded the 'features' and 'labels' pickled data for the training, validation and testing set. I calculated the summary of the data set using **numpy**. I created a **pandas** dataframe out of the data summary and then visualized the dataframe in a table using **matplotlib**:

**Table 1: GTSRB Dataset Summary**

![data_summary.jpg](assets/data/images/output/data_summary.jpg)

### Exploratory Visualization

I performed multiple exploratory visualizations on the German Traffic Sign Dataset using **csv** and **matplotlib**. First, I displayed the 43 German Traffic Signs by image and traffic sign type in subplots as a gallery:

![traffic_sign_gallery.jpg](assets/data/images/output/traffic_sign_gallery.jpg)

**Figure 1: Gallery of Traffic Sign Types**

I displayed the count of occurrences for each unique traffic sign in a table for the training set, validation set and testing set:

**Tabe 2: Count of Traffic Sign Type Occurrences**

![count_traffic_sign_reps.jpg](assets/data/images/output/count_traffic_sign_reps.jpg)

Similary, I visualized the distribution of traffic sign types in the training, validation and testing set as individual bar graphs:

![distr_sign_train_set.jpg](assets/data/images/output/distr_sign_train_set.jpg)

**Figure 2: Distribution of Traffic Sign Types in Training Set** 

![distr_sign_valid_set.jpg](assets/data/images/output/distr_sign_valid_set.jpg)

**Figure 3: Distribution of Traffic Sign Types in Validation Set** 

![distr_sign_test_set.jpg](assets/data/images/output/distr_sign_test_set.jpg)

**Figure 4: Distribution of Traffic Sign Types in Testing Set** 

From looking at **Figure 2**, the distribution bar graph for traffic sign types in the training set, **keep right**, **yield**, **priority road**, **speed limit (50km/h)** and **speed limit (30km/h)** have highest occurrences. Therefore, when the model is trained, it probably will do well in classifying those particular traffic sign types. Since the validation set has a similar distribution for traffic sign types as the training set, the model will most likely achieve a high validation accuracy. Likewise, the testing set also has this similar distribution for traffic sign types as the training set, so the model's predictions will probably be high testing accuracy too. However, if I later download German Traffic Signs from the internet that the model did not learn well and have the model classify them, the model's predictions for these new web images will most likely be low accuracy.

## Design and Test a Model Architecture

### Preprocessing

Data preprocessing techniques of data augmentation, image normalization, data shuffling and one-hot encoding were performed with **OpenCV** and **sklearn** to improve the deep learning classification model's ability to learn to classify traffic signs, resulting in higher prediction accuracy.

#### Data Augmentation

Data augmentation was applied to the training set using **OpenCV** since there was not a sufficient amount of data for the model to generalize well. From looking at the bar graph distribution of traffic sign types for training set in **Figure 2**, we could see the data is unbalanced and some traffic sign types are represented to a higher extent than other types. We will apply rotation operations to augment the data and work on balancing the training set.

After experimenting with scaling, translation, rotation, flipping, salt and pepper noise and lighting condition augmentation techniques, I used **rotation transformations** in the Python pipeline. I applied 6 rotation operations from -15 to +15 degrees to augment the training set, so the model can recognize the traffic signs in multiple orientations. There was not a need to rotate images past plus or minus 15 degrees because traffic signs are typically perpendicular to the ground and sometimes we may find them slightly tilted.

![augmented_rotation_data.jpg](assets/data/images/output/augmented_rotation_data.jpg)

**Figure 5: Augment Image with 6 rotations** 

After rotation augmentation was applied to the training set, it increased from 34,799 to 208,794 images.

#### Image Normalization

[Min-Max scaling](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html#about-min-max-scaling) normalization was applied to training, validation and testing data to change the range of pixel intensity, so all images can have a consistent range for pixel values. 

Image normalization was chosen as a preprocessing technique to deal with contrast stretching, stabilize the model and improve gradient calculation.

Equation:

![X' = a \pm ((X - X_{min})(b-a)/(X_{max} - X_{min}))](https://latex.codecogs.com/svg.latex?X%27%20%3D%20a%20%5Cpm%20%28%28X%20-%20X_%7Bmin%7D%29%28b-a%29/%28X_%7Bmax%7D%20-%20X_%7Bmin%7D%29%29)

### Data Shuffling

Data shuffling was performed using **sklearn** on the entire training set, so the training images in each batch can better represent the entire training set.

### Model Architecture

I implemented a 5 layer CNN model based on [Yann LeCun's LeNet model architecture presented in his 1998 research paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) using Tensorflow. The architecture consists of 2 convolution layers followed 3 fully connected layers. The convolution layers handle extracting features from the traffic sign images, which are useful for learning how to recognize them. Each convolution layer is followed by relu activation and max pooling for filtering images down to the pixels that matter. After the last convolution layer, flatten operation is applied to transform the output shape into a vector. The first two fully connected layers are followed by relu activation and dropout for preventing the model from overfitting the data. The fully connected layers shrink the output vector to 43 different classes, so the model can predict the traffic sign type in the image. The code implementation for our modified LeNet architecture can be found under the **LeNet()** function in the Jupyter notebook. For visual reference of the LeNet model architecture, view the following diagram. Our version of LeNet will have some minor differences and will be explained after the diagram:

![lenet_mod](assets/data/images/lenet_mod.jpg)

**Figure 6: Modified LeNet-5 Architecture**

The breakdown of our modified lenet-5 architecture will be discussed below:

Here are the formulas for calculating convolution and max pooling output height and width.

**Formula for Convolutions**:

![Out_{h} = (in_{h} - filter_{h} + 1)/strides[1]](https://latex.codecogs.com/svg.latex?Out_%7Bh%7D%20%3D%20%28in_%7Bh%7D%20-%20filter_%7Bh%7D%20&plus;%201%29/strides%5B1%5D)

![Out_{w} = (in_{w} - filter_{w} + 1)/strides[2]](https://latex.codecogs.com/svg.latex?Out_%7Bw%7D%20%3D%20%28in_%7Bw%7D%20-%20filter_%7Bw%7D%20&plus;%201%29/strides%5B2%5D)

**Formula for Max Pooling**:

![Out_{h} = ((in_{h} - filter_{h})/S)+1](https://latex.codecogs.com/svg.latex?Out_%7Bh%7D%20%3D%20%28%28in_%7Bh%7D%20-%20filter_%7Bh%7D%29/S%29&plus;1)

![Out_{w} = ((in_{w} - filter_{w})/S)+1](https://latex.codecogs.com/svg.latex?Out_%7Bw%7D%20%3D%20%28%28in_%7Bw%7D%20-%20filter_%7Bw%7D%29/S%29&plus;1)

The network takes a 32 by 32 image as input, then that image goes through convolutional layer 1.

Convolutional layer 1 has a 5x5 filter with an input depth of 3 and an output depth of 64 and initilizes the bias. Then this layer convolves the filter over the images and adds the bias at the end giving us an output of 28 by 28 by 64. The **formula for convolutions** tell us how to calculate the output height and width for convolutional layer. Next the output of this layer is activated with a relu activation function. Then the output is pooled with a 2 by 2 kernel with a 2 by 2 stride giving us a pooling output of 14 by 14 by 64. The **formula for max pooling** tells us how to calculate the output height and width for max pooling.

The network then runs through convolutional layer 2, which is a set of convolutional, relu activation and pooling layers giving an output of 5 by 5 by 32.

Then this output is flattened to a vector with length 5x5x32 equalling 800. 

This vector is passed to fully connected layer 3 with a width of 120. Then a relu activation function is applied to the output of this fully connected layer. Finally, the dropout regularization technique is performed on the output dropping 50% of the neurons. Dropout is used to minimize overfitting to the training data and improve generalization of the model.

The network then runs through fully connected layer 4, which is a set of fully connected, relu activation and dropout layers giving an output width of 84.

Finally, fully connected layer 5 is connected to the end of the network to output a width equal to the number of traffic sign types in our label set, 43 classes.

### Model Training

First I setup Tensorflow hyperparameters and variables used for training the model:

The **learning rate equals 0.001** since it is a good default value. The learning rate tells Tensorflow how quickly to update the weights. The **epochs equals 10** to tell Tensorflow to run the training data through the network 10 times. The epochs was set to 10, so training would not take too long and because the model's validation accuracy was near 100%. The **batch size equals 64** to tell Tensorflow to run 64 training images at a time. I decided to go with a smaller batch size to consider processors with memory limitations. The downside of using 64 batches is that the model training will take longer, but that should not be a problem since the epochs equals 10.

Now I need to setup the Tensorflow variables. **x** is a placeholder that stores the input batches. The batch size for x is initialized to None to allow the placeholder to later accept a batch of any size. The image dimensions for x were set to 32 by 32 by 3. **y** is a placeholder that stores the labels as integers, meaning they are not one hot encoded. The **tf.one_hot(y, 43)** function is used to one hot encode the labels and that result is stored into **one_hot_y**.

Now with the Tensorflow hyperparameters and variables setup, I can activate the training pipeline by passing training images to it. The pipeline then trains the model. Let's dive into the pipeline to see how it trains the model. As data is pushed into the pipeline, that data is loaded into the **LeNet()** function to calculate the logits. Those logits are then compared with the ground truth labels and the cross entropy is calculated using the **tf.nn.softmax_cross_entropy_with_logits()** function. The average cross entropy from all the training images is computed using the **tf.reduce_mean()** function. Next the Adam optimizer with the learning rate minimizes the loss function using the **tf.train.AdamOptimizer()** function. Stochastic gradient descent is similar to the Adam optimizer, but we used Adam because it is more sophisticated. Finally the pipeline runs the **minimize()** function on the optimizer, which uses backpropagation to update the network and minimize the model's training loss.

With the training pipeline implemented, let's explore how the 64 batches of training data was being passed to the pipeline to train the model. A Tensorflow session was created to initialize the Tensorflow variables and train the model over 10 epochs. At the start of each epoch, the training data was shuffled to ensure that the training data is not bias by the order of the images. Next the training data was split into batches, the batch data was passed to the training pipeline and the model was trained on each batch. 

How do we know if our trained model has high prediction accuracy and is worth saving? We need to evaluate the model while it is being trained.

#### Model Evaluation

Now that the model is trained, an evaluation pipeline was used to evaluate the model's performance. Let's dive into the pipeline to see how it evaluates the model. The first step in this pipeline was to compare the logit prediction to the one hot encoded ground truth label using **tf.equal(tf.argmax(logits,1), (tf.argmax(one_hot_y,1))**. The last step in this pipeline was to calculate the model's overall accuracy by averaging the individual prediction accuracies using **tf.reduce_mean(tf.cast(correct_prediction, tf.float32)**.

With the evaluation pipeline implemented, an **evaluate()** function was used to actually run the evaluation pipeline. This function takes a dataset as input, batches the dataset, then runs it through the evaluation pipeline and averages the accuracy of each batch to calculate the total accuracy of the model.

With the **evaluate()** function implemented to run the evaluation pipeline, let's return to the stage of training our model, at the end of each epoch, we can also evaluate the model's prediction accuracy on the validation data. Once we have completely trained the model and it achieves high prediction accuracy at least 93% on the validation data, we can save the model and later test it on the testing data.

### Solution Approach

My final model results for traffic sign recognition were **Validation Accuracy = 98.4%** and **Test Accuracy = 96.5%**. When starting the project, I chose the LeNet architecture developed by Yan Lecun in 1998. This network architecture was already well known for it's great performance on image recognition, specifically on classifying hand written numbers from the MNIST dataset. Since hand written digit recognition and traffic sign recognition are similar, both use cases detect patterns in images and match them with labeled training data, this model with some modifications would work well for my project.

When I trained and evaluated the model for the first time, the result was 89.5% validation accuracy. To meet the requirements for the project, the validation accuracy needed to be 93% or higher. I tried tuning the hyperparameters for the model and found changing the batch size to 64 increased validation accuracy. I also found that adding the dropout regularization layers at the end of the first two fully connected layers increased the validation accuracy to 95.1%. Once I started augmenting the training data with 6 rotations from -15 to +15 degrees followed by normalizing the data, then I noticed the validation accuracy increase to 98.4%.

## Test a Model on New Images

### Acquiring New Images

I downloaded 5 German Traffic Signs of varying sizes from the internet. I did not consider whether the image size was the same size I used to train the model. I also chose some images where the traffic signs were not at the center of the image like images used in the training, validation and testing set. Some of the images have cars, buildings and trees in the background. I was curious to see how accurate the trained model would be at classifying traffic signs with more noise than usual.

![web_test_images.jpg](assets/data/images/output/web_test_images.jpg)

**Figure 7: New Images**

I preprocessed the images by resizing them to 32 by 32, so they would be the dimensions the model expects. As one looks at the following image, the features look distorted compared to the original image due to resizing, which may cause the model's prediction accuracy to be lower.

![resized_web_test_images.jpg](assets/data/images/output/resized_web_test_images.jpg)

**Figure 8: Resized New Images**

After resizing the image, I normalized the new images before passing it to the model to be predicted.

### Performance on New Images

The model classified 3 out of 5 traffic signs correctly for the new images from the internet. Thus, the model's test accuracy was 60%. When comparing the model's prediction accuracy on the new images to the accuracy results on the testing set, the model performed 36% worse. This performance accuracy makes sense since these 5 new images had different sizes, rotations, noise and lighting conditions compared to the testing images in the German Traffic Sign Dataset.

I used **tf.argmax()** function and passed logits to it to retrieve the predicted traffic sign type numbers. Earlier we mentioned, these traffic sign type numbers can be used in the **signnames.csv** file to look up the traffic sign names.

![predict_sign_type.jpg](assets/data/images/output/predict_sign_type.jpg)

### Model Certainty - Softmax Probabilities

For the first image, the model was **53.3%** certain that the image was a traffic sign indicating "**Children crossing**," but the traffic sign in the image was actually a "**Speed limit (50km/h)**." All the remaining softmax probabilities for this first image were not correct. For the second image, the model was **100%** certain the image contained a traffic sign indicating "**Traffic signals**," but it was actually a "**Children crossing**" sign. For the third, fourth and fifth image, the model was 100% certain that the image was a "**No entry**," "**Priority road**" and "**Road work**" traffic sign. For the last 3 new images, the model was **100%** certain in it's prediction and it was also correct.

![top5_softmax_probabilities.jpg](assets/data/images/output/top5_softmax_probabilities.jpg)

### Visualize Layers of the Neural Network

Deep neural networks are often referred to as a black box. We can understand what the weights of a network look like by plotting the network's feature maps. After the network was trained, I passed a test image and the name of a convolutional layer into an **outputFeatureMap()** function to visualize that layer's feature maps. The feature map shows the patterns that the network's layer is detecting. Here is an example of visualizing convolutional layer 2 pooling **conv2_pool**, notice there are 32 feature maps:

![conv2_pool.jpg](assets/data/images/output/conv2_pool.jpg)

## Conclusion

This project focused on building a traffic sign recognition application using exploratory data analysis (numpy, pandas, matplotlib), computer vision (OpenCV) and deep learning (Tensorflow). During this project, I learned to interpret a deep neural network architecture from a research paper and implement it with Tensorflow. I learned to visualize all sorts data to bring meaningful insight to users and increase the model's prediction accuracy using data preprocessing techniques and tuning the model's hyperparameters. I would highly recommend this project for anyone who wants to gain practical experience with Tensorflow, data analysis and deep learning.

## Udacity Self-Driving Car Engineer ND References:

- Udacity SDCE ND, Lesson 12 - Tensorflow
- Udacity SDCE ND, Lesson 13 - Deep Neural Networks
- Udacity SDCE ND, Lesson 14 - Convolutional Neural Networks
- Udacity SDCE ND, Lesson 15 - LeNet for Traffic Signs

## How to References

The following covers examples of visualizing one or more images in a figure using Matplotlib subplots. 

- [Creating multiple subplots using plt.subplots](https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html)

The following link covers how to use numpy unique() function. This function was helpful for counting the number of times a traffic sign appears. It also helped me find the index of the first occurrence of a new traffic sign:

- [NumPy Array manipulation: unique() function](https://www.w3resource.com/numpy/manipulation/unique.php)

The following link covers different data augmentation techniques:

- [Data Augmentation | How to use Deep Learning when you have Limited Data - Part 2](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/)

The following link covers how to use Keras ImageDataGenerator for data augmentation:

- [Keras ImageDataGenerator and Data Augmentation](https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/)

The following link covers data augmentation techniques in CNN using Tensorflow:

- [Data Augmentation Techniques in CNN using Tensorflow](https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9#c3f1)

The following link covers how to rotate images with OpenCV:

- [Rotate images (correctly) with OpenCV and Python](https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/)

The following link covers how to rotate image 90, 180, 270 degrees using OpenCV:

- [OpenCV Python 90, 180, 270 - Example](https://www.tutorialkart.com/opencv/python/opencv-python-rotate-image/)

The following link covers image translation using OpenCV:

- [Image Translation using OpenCV Python](https://www.geeksforgeeks.org/image-translation-using-opencv-python/)

The following link covers geometric transformations on images using OpenCV:

- [Geometric Transform of Images](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html)

The following link covers how to add salt & pepper noise to your images using OpenCV:

- [Salt & Pepper Noise and Median Filters, Part II - The Code](https://blog.kyleingraham.com/2017/02/04/salt-pepper-noise-and-median-filters-part-ii-the-code/)

The following link covers how to change the contrast and brightness of an image using OpenCV:

- [Changing the contrast and brightness of an image!](https://blog.kyleingraham.com/2017/02/04/salt-pepper-noise-and-median-filters-part-ii-the-code/)

## Traffic Sign Classifier Article References

- [DRIVE Labs: Classifying Traffic Signs and Traffic Lights with SignNet and LightNet DNNs](https://news.developer.nvidia.com/drive-labs-signnet-and-lighnet-dnns/)

- [Traffic Sign Detection and Classification in the Wild](https://zpascal.net/cvpr2016/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf)

- [Term1-P2: Traffic Sign Classifier Project - Part 1](https://medium.com/@gongf05/term1-p2-traffic-sign-classifier-project-f011ed053f8)

- [Traffic signs classification with convolutional network](https://navoshta.com/traffic-signs-classification/)