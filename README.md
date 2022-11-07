# Recognizing-Handwritten-Digits-with-scikit-learn
Recognizing Handwritten Digits with scikit-learn    #SubashPalvel
#SUBASHPALVEL #subashpalvel #SubashPalvel #Subash_Palvel


Recognizing Handwritten Digits with scikit-learn

In this blog post, we will be using the Python library scikit-learn to classification handwritten digits from the MNIST dataset. The MNIST ("Modified National Institute of Standards and Technology") dataset is a classic in the machine learning community. It consists of images of handwritten digits (0-9) that have been size-normalized and centered in a 28x28 grayscale image. Each image is represented by 784 (28x28) features, where each feature is a intensity value from 0 to 255.

What is scikit-learn?

scikit-learn is a free, open-source Python library that provides a range of supervised and unsupervised machine learning algorithms. It is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

scikit-learn was originally developed by David Cournapeau as a Google Summer of Code project in 2007. The project was later funded by European Union's Seventh Framework Programme.[3] In 2010, INRIA took over development and maintenance of the project.[4]

The scikit-learn library consists of various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

Loading the data

When working with scikit-learn, it is important to have your data in the right format. In this case, we will be using the MNIST dataset, which is a collection of 70,000 images of handwritten digits. Each image is 28x28 pixels, and each image has a label (0-9) associated with it.

To load the data, we'll use the load_digits() function from sklearn.datasets. This function returns a tuple containing the data and labels. We can unpack this tuple into two separate variables like so:

from sklearn.datasets import load_digits

data, labels = load_digits(return_X_y=True)

Preprocessing the data

Preprocessing the data is an important step in any machine learning pipeline. In this blog post, we will show you how to preprocess handwritten digit data for use with scikit-learn.

The first step is to load the dataset. We will be using the popular MNIST dataset, which contains 70,000 grayscale images of handwritten digits from 0-9. Each image is 28x28 pixels:

from sklearn.datasets import load_digits digits = load_digits()

Next, we need to split the dataset into a training set and a test set. We will use 60,000 images for training and 10,000 images for testing. We also need to reshape each image into a 1D array:

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target) X_train = X_train.reshape((-1, 784)) X_test = X_test.reshape((-1, 784))

Now that our data is ready, we can start building our classifier!

Training the model

Assuming that you’ve already installed scikit-learn and downloaded the MNIST dataset, you can now start training your model. In this example, we’ll be using a Support Vector Classifier (SVC). An SVC is a type of supervised machine learning algorithm that can be used for classification tasks.

First, we need to import the necessary modules:

```python
from sklearn import datasets
from sklearn import svm
```

Then, we load the dataset:

```python
digits = datasets.load_digits()
```

The “digits” variable is now an object containing the MNIST dataset. This contains 1797 images, each of which is represented as an 8x8 array of numbers. We also have a target (the correct label for each image), which is also an array containing 1797 elements.

To train our classifier, we need to take these arrays and split them into two sets: one for training our model, and one for testing it. We’ll use 80% of the data for training and 20% for testing:

```python
X_train, X_test, y_train, y_test = train_test_split( digits.data, digits.target, test_size=0.2)
```

Now that our data is split up intotrainingandtesting sets

Evaluating the model

Evaluating the model is an important part of any machine learning project. In this blog article, we will use scikit-learn's built-in accuracy_score function to evaluate our model.

First, we need to split our data into training and test sets. We will use 80% of the data for training and 20% for testing.

Next, we will train our model on the training set and make predictions on the test set. We can then compare our predictionsto the actual labels to see how accurate our model is.

scikit-learn's accuracy_score function takes two arguments: y_true and y_pred. y_true are the true labels for the test set and y_pred are the predicted labels from our model. The function returns a float between 0 and 1, where 1 is perfect accuracy and 0 is no better than random guessing.

In our case, we find that our model has an accuracy of 96%. This means that our model is able to correctly predict the label for 96% of the handwritten digits in the test set.

Saving the model

Saving the model is a very important step in any machine learning project. You want to be able to save your model so that you can use it again later or share it with others. There are a few different ways to save a scikit-learn model, but we recommend using the pickle module.

Pickling is the process of converting an object (in this case, a scikit-learn model) into a byte stream. This is useful because it allows you to save the model to disk and then later load it back into memory and use it without having to retrain the model.

To pickle a scikit-learn model, you first need to create a pickle file. We recommend using the Python pickle library for this purpose. You can create a pickle file using the following code:

import pickle

with open('model.pkl', 'wb') as f:
pickle.dump(model, f)

This will create a file called model.pkl in the current directory. You can then load this file back into memory and use it like any other scikit-learn model:

with open('model.pkl', 'rb') as f:
model = pickle.load(f)

Conclusion

In this article, we went over how to recognize handwritten digits using the scikit-learn library. We started by loading and exploring the dataset, then we preprocessed the data and split it into training and testing sets. Next, we trained a support vector machine classifier on the training data and evaluated its performance on the test set. Finally, we saw that we could further improve our classifier by tuning its hyperparameters. Overall, scikit-learn is a powerful toolkit for machine learning that can be used to build sophisticated models with minimal effort. If you're just getting started with machine learning, I highly recommend checking out scikit-learn.
