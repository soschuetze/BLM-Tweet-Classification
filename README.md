# BLM-Tweet-Classification

This project explores the research question of which natural language processing techniques and machine learning models best classify the forms of activism contained in #BlackLivesMatter tweets.

## Table of Contents
* [General Info](#general-information)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)


## General Information
The labels include:

0 = Unrelated to #BlackLivesMatter

1 = Within the system calls for action (e.g. voting, contacting an elected official, etc.)

2 = Disruptive calls for action (e.g. protesting, boycotting, etc.) 

3 = Raising awareness/spreading information (e.g. retweet, like, spread the word) 

4 = Other (symbolic, buy swag, etc.) 

5 = Moral encouragement/support

6 = Community gatherings for solidarity (vigil, community concert, etc.) 

7 = Oppositional actions 

8 = Pressuring non-political elites (e.g. media, advertiser)


The project tests preprocessing methods (stop word removal, stemming, and lemmatizint), word embdedings methods (tf-idf and sBert), and various models (perceptrons, knn, svm, neural networks, and fine-tuning distilbert).

## Setup
Install the following libraries:
transformers
tensorflow
pandas
numpy
scikit-learn
matplotlib
nltk
sentence-transformers
pytorch
tqdm

## Usage
Download all files and keep in the same folder.

1. Run clean_tweets.py file using the supervised_tweets csv - this python file runs the code that will replace all links in the tweets with the word "URL" and lowercase all words. This file will create a new csv (supervised_clean) with the cleaned tweets in the same directory that you're running the python file.
The next files can be run in any order.
2. supervised_methods.py - this file runs all of the sklearn machine learning methods (perceptron, knn, svm, and neural network) with the chosen preprocessing steps and outputs the mean cross validation accuracy, recall, precision, and F1-scores. To choose which preprocessing steps to use, uncomment your choice of lines 147-149 – these respectively correspond to removing stop words, lemmatizing, and stemming. All of the metrics will be printed to the console.
3. distilbert_based_uncased.py - this file fine-tunes the distilbert model. It will print the results of the epochs and create a confusion matrix.
4. training_neural_network.py - this file trains the neural network which uses the pretrained distilbert base uncased model again and its associated tokenizer. After the distilbert layer there will be four linear regression layers, a flattening layer, and a final linear regression layer before applying softmax to classify each point. 

## Project Status
Project is: in progress

## Room for Improvement
To do:
- Try multilabel classification methods
- Combine labels

