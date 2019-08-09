# Network Intrusion Detection with Feature Extraction ML



# Network Intrusion Detection
The project detects the connection is good or bad based on the various network parameters like connection protocol, destination IP,  souce IP and many more features in the original dataset.
# 1.Problem Statement
This project aims to detects good or bad connection to the network featuing binary classification '0' for good connection and 1 for bad connection. 

**Feature Extraction | Feature Importance Analysis | Feature Selection | Pearson's Coefficient | RFE | Standard Scaler | l1 -l2 Regularization | Pertubation Rank**

Models : Logistic Regression | Nearest Neighbor | Support Vector Machine | Gaussian Naive Bayes | Fully-Connected Neural Networks | Convolutional Neural Networks (CNN)

Project attemps to learn:
* Feature Extraction
* Feature Selection and Feature Importance Analysis
* Apply various Feature importance / selection algorithm like Pearson's coefficient, RFE, Extra Tree Classifier etc.
* Dealing with the huge dataset with having more features/columns
* Compare the model performances before and after applying any feature selection technique/feature extraction
* l1-l2 regularization for Neural Networks
* Visualizing any data to 4D image to feed in the CNN model
* Binary classification converted into multi-classification for comparison
* ROC curve & Confusion Matrix

# 2. Dataset
This database has a wide variety of intrusions collected in a military network environment.

* Preprocessed the data, dropped the rows with null values, unnecessary columns
* Normalized numeric data with zscore normalization
* Labeled the ouput feature
* One hot encoded the categorical data

# Splitting the data 
* Spliting the data into 75% train - 25% test

# Without Feature Extraction
* With all 42 columns fed to the model (117 after normalization & one-hot encoding & labeling)
* Tried with different parameters for each model

## Logistic Regression
## KNN
## SVM
## GNB
## Fully Connected Nueral Network
## CNN

# With Feature Extraction For every model
## Feature Importance Analysis
## Pearson's Coefficient Correlation
* The Pearson's Coefficient Correlation shows how correlated the output column is with every other feature in the dataset.

* Thus filters out which features are more corelated to the output. (117 features)
* Selected the features to a certain threshold and split the data in to 75-25 for train - test
## LR
* Tried only with PCC and applied RFE on top of it to still observe the difference

## LR with RFE

## KNN
* Best K value

## SVM
* Tried with best C value
## GNB
## GNB with Feature Importance
* Used Extra Tree Classifier and less features than above
* Got almost same result
## Fully Connected Nueral Network
* Tried several models with parameter tuning used checkpoints and earlystopping
* Tried with Standard Scaler on KNN, turns out no imptovement
## feature importance
* Calculated Pertubation rank for tried features
* Got the almost same accuracy with less features
## CNN
## CNN with l1 l2 regularization technique
# Converting Binary to Multi classification Problem
* Classifying the attacks into 4 main category of the intrusion which are DoS, Probe, U2L and R2L.
* Again having 40 columns initially - 117 after normalizing, labelling and one hot encoding
## Select Percentile and RFC
* Used Select Percentile & fclassif to a certain threshold and chose features
* On those feature Random Forest Classifier was applied
* Spliting the data into train-test as 75-25
## Logistic Regression
## KNN
## SVM
## GNB
## Fully Connected Nueral Network
## CNN
# Comparison
# Conclusion
* While dealing with the big data the feature selection plays a vital role thus understanding the dataset field and preprocess it in the correct format is very important.
* Acuuracy before and after applying the feature extraction remains almost same that means choosing the feature is crucial for ML model.
* Binary and multi-class classification F1score does not have any significant difference
