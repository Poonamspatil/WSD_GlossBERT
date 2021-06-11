# Image Classification using Statistical Modeling, CNN and Deep neural network architecture (InceptionV3, ResNet-50)

## Table of contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [Description](#description)
* [Results](#results)

## Introduction

Among the most well-known tasks associated with computer vision is image  classification. Where, images are classified as belonging to one of a set of predefined categories. Objective of this notebook is to explore and implement computer vision (CV) techniques (with logistic regression, deep learning neural networks, transfer learning etc.) to identify and process images like human vision. 

### Image classification applications
Applications where image classification plays a vital role.
1. Medical Diagnosis
Image classification can be used in detecting diseases for example predicting Covid-19 cases from chest X-ray images (binary classification).
2. In the applications of Computer Vision (CV), solving problems related to image classification plays an important role for example assigning a category to a photograph of a face (multiclass classification) etc.

### Project Link
Knowledge Competition from August 2019 ([link](https://zindi.africa/competitions/sbtic-animal-classification))
Study Objective: The study of these images aims to investigate herbivore coexistence for which image classification is a crucial step to identify zebra from elephants. [Link](https://www.zooniverse.org/projects/zooniverse/snapshot-serengeti/about/research)

### Data Source
Data repository for the University of Minnesota ([link](https://conservancy.umn.edu/handle/11299/199819))
The data takes 18,000+ images of zebras and elephants
Training set : Given 13,999 images	Test Set: 5000 images
Image resized to 330x330 pixels in original data.

## Description
This project demonstrates different classification models for the task and compares the results. 
Using different libraries for the task is also demonstrated. 
The data is divided in train/valid and test split.
The notebook is consists of mainly in 2 parts. 
1. Part 1: employs statistical image pre-processing techniques in one way and thereby evaluates results on 4 classifiers- Logistic, kNN, Random Forest and SVM classifiers. 
2. Part 2: employs deep learning approaches and uses CNN and Transfer Learning approach (with ResNet50,InceptionV3).

### General Approach to Image classification

1. Data Loading
2. Data preprocessing (read image, resize image, data augmentation)
3. Different Machine Learning techniques for the task at hand. 

## Technologies
Project is created with:
* Python : 3.7
* tensorflow: 2.3.0

## Results
Accuracy metrics is used for comparing performance of different models. Logistic Regression classifier is taken as baseline model.  
Best results were achieved with transfer Learning from a pre-trained model InceptionV3.