# Predicting_Malignant_Cell_growth
This repository contains the files needed to use supervised machine learning to predict cancer related malignant cell growth. This is done using six different supervised machine learning classifiers (elastic net, KNN, CART, SVM, bagged CART, random forest) which will classify the sample as either benign (B) or malignant (M). Each classifier is tuned according to its own parameter and the final results (specificity, sensitivity, AUC) are summarized in a table. The data used to create the testing and training datasets can be obtained from http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic). 

predict.R: This file contains all the commented R code for each of the six classifiers used. The file contains a summary of each classifier alognside the parameter that will be tuned for optimization. 

trainset.csv: This csv file contains the data that is used to train and optimize each of the classifiers

testset.csv: This csv file contains the data that is used to test each of the trained models
