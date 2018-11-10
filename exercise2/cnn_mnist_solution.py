from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten

# class to keep track of accurancy
class ConvNetHist(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc =[]

    def on_epoch_end(self, batch_size, logs={}):
        self.acc.append(logs.get("acc"))


def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)

def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr, num_filters, filter_size, batch_size):
    # define input shape of sigle image
    input_shape = (28, 28, 1)
    # clear the session , necessary for training with different hyper parameters
    keras.backend.clear_session()
    # create the model
    model = Sequential()
    # add layers to the model
    model.add(Conv2D(num_filters, kernel_size=(filter_size, filter_size), padding="same", activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(num_filters, kernel_size=(filter_size, filter_size), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # flattn for input in fully connected layer
    model.add(Flatten())
    # add fully connected layer and softmax layers
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    # compile model with categorical_crossentropy and stochastic gradient descent
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=lr), metrics=["accuracy"])
    # See ConvNetHist class
    history = ConvNetHist()
    # fit model to train data and test on validation data
    model.fit(x_train, y_train, batch_size=batch_size,
        epochs=num_epochs, verbose=1, validation_data=(x_valid, y_valid), callbacks=[history])
    # store learning curve from ConvNetHist
    learning_curve = history.acc
    # calculate the error for all elements in learning_curve
    learning_curve[:]= [1 - x for x in learning_curve]
    return learning_curve, model


def test(x_test, y_test, model):
    # evaluate the model on the test data set
    result = model.evaluate(x_test, y_test, verbose=0)
    # needs to be done as result is accuracy
    test_error = 1 - result[1]
    return test_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=0.1, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=16, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=15, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")

    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs


    # Exercise 2
    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)
    plt.figure(figsize=(13, 10))
    plt.xlabel("epoch")
    plt.ylabel("error")
    for lr, color in [[0.1,"green"], [0.01, "blue"], [0.001, "red"], [0.0001, "black"]]:
        learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters,3, batch_size)
        plt.plot(range(1,epochs + 1),learning_curve, label="learning_rate: %1.4f" %lr,linestyle='-',color=color)
    plt.legend(loc="best", prop={'size': 15})
    plt.savefig('lr_variations.svg')


    # Exercise 3
    plt.figure(figsize=(13, 10))
    plt.xlabel("epoch")
    plt.ylabel("error")
    for filter_size, color in [[1,"green"], [3, "blue"], [5, "red"], [7, "black"]]:
        learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters,filter_size, batch_size)
        plt.plot(range(1,epochs + 1),learning_curve, label="filter_size: %i" %filter_size,linestyle='-',color=color)
    plt.legend(loc="best", prop={'size': 15})
    plt.savefig('filter_size_variations.svg')

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["learning_curve"] = learning_curve
    results["test_error"] = test_error


    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()
