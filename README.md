# X-Ray-Detection-CNN_Model-
This is a model which can do  multi image classification for 4 classes Covid,Lung_Opacity,Viral Pneumonia,Normal via X Ray Images and its Masks.The dataset used here is COVID-19 Radiography Database from Kaggle.The link for the dataset is https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database


In this model it uses a four layered custom trained CNN layers for detecting images first then while training each image has a masked layer so in the X ray scan the unwanted regions are removed by the mask and as a 2 layered image,mask is used in training for the multiclass image detection and it also introduces data augumentation for lesser imbalanced classes available in the dataset which is of two types strong augumentation and normal auguentation.The stronger augumentation suggests that the classes with less images/masks are taken care with more priority with the more number of images in other class via normal augumentation layer.


This model training got around a 90% accuracy on the validation set consisting of 100 images of each class making total of 400 images with masks.While the dataset was splitted into train/validate set via the other python code available in the repository.Other code is used for training the model and displaying the results in form of accuracy,loss score,f1 score and confusion matrix each plotted visually.The third code is also used to validate the dataset taken from original dataset
