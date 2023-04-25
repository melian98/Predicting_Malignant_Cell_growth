#make a vector of the packages to be installed and loaded
packages <- c("tidyverse", "caret", "glmnet", "class", "ggthemes", "tree", "e1071", "pROC", "randomForest")

#make a vector of any packages that are not installed
missing_packages <- packages[!(packages %in% installed.packages()[ , "Package"])]

#if any packages are not installed, they will be installed
if(length(missing_packages)) {install.packages(missing_packages)}

#load all of the packages
lapply(packages, require, character.only = TRUE)

#remove unneeded variables
rm(list = ls())

#load in the csv and generate additional columns which will square each column to account for the quadratic effects.
training <- read_csv("trainset.csv")
temp <- (training[ , 2:ncol(training)])^2
colnames(temp) <- c(paste0(rep("X", 30), 31:60))
training <- cbind(training, temp)

validation <- read_csv("testset.csv")
temp <- (validation[ , 2:ncol(validation)])^2
colnames(temp) <- c(paste0(rep("X", 30), 31:60))
validation <- cbind(validation, temp)


testing <- validation[ , 2:ncol(validation)]

#create a final results dataframe which will contain the results from each model for later comparison
final_results_df <- as.data.frame(matrix(nrow = 1, ncol = 7))
colnames(final_results_df) <- c("model", "variable", "value", "Sensitivity", "Specificity", "difference", "AUC")

#set seed to ensure reproducibility
set.seed(976)

#remove unneeded variables
rm(temp)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# model: Elastic net (including LASSO and ridge regression)
# 
# Description: This section includes the code for the elastic net supervised machine learning model. It will compute the Sensitivity, Specificity and AUC for 
# Ridge, LASSO and elastic net regression. Elastic net will be filtered for the parameter values that yield the best model. The best model will be chosen as the
# model that has the lowest difference between its sensitivity and specificity
# 
# Parameter: The parameter that will be changed for tuning in this model is the alpha value for elastic net classification. The alpha value will be incremented
# by 0.01 and the best alpha values will be decided based on the sensitivity and specificity that is obtained.
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#elastic_net function will be called to perform the calculations for the models
elastic_net <- function(type) {
  
  #set alpha equal to the appropriate value depending on the model being used or return an error if an inappropriate model was selected
  if (type == "ridge") { alpha = 0}
  else if (type == "lasso") { alpha = 1 }
  else if (type == "elastic") {alpha = 0.5}
  else {
    cat("Please enter either \"ridge\" or \"lasso\" or \"elastic\"")
    return("error")
    } 
  
  #if ridge or LASSO regression are being used 
  if (alpha == 0 || alpha == 1)
  {
    #train the model using the alpha value corresponding to ridge or LASSO regression
    model <- cv.glmnet(x = data.matrix(training[ , 2:ncol(training)]), y = as.factor(training$status), type.measure = "mse", alpha = alpha, family = "binomial", keep = TRUE)
    
    #use the model to classify the testing dataframe
    predict <- predict(object = model, s = model$lambda.min, newx = data.matrix(testing), type = "class")
    
    #compute the sensitivity and specificity in a confusion matrix
    confusion_matrix <- confusionMatrix(data = as.factor(predict), reference = as.factor(validation$status))
    
    #change the predictions into an ordered factor
    predict <- factor(predict, ordered = TRUE)
    
    #calculate the roc_line to visually demonstrate the models performance
    roc_line <- roc(predictor = predict, response = as.factor(validation$status))
    
    #print an ROC plot comparing the sensitivity and specificity at different values
    print(ggroc(roc_line, colour = "steelblue", size = 2) +
      ggtitle(paste0("ROC for ", type, " regression")) +
      geom_abline(slope = 1, intercept = 1))
    
    #calculated the AUC for model evaluation
    auc <- auc(validation$status, predict)
    
    #calculate the absolute difference between the sensitivity and specificity for model evaluation
    difference <- abs(confusion_matrix$byClass["Sensitivity"] - confusion_matrix$byClass["Specificity"])
    
    #return all model evaluation parameters to be added to the final_results_df
    return(c(type, "alpha", alpha, confusion_matrix$byClass["Sensitivity"] , confusion_matrix$byClass["Specificity"], difference, auc))
  }
  
  #if elastic net is chosen (alpha is not equal to 0 or 1)
  else {
    
    #a results dataframe containing all alpha values that display the best results and their sensitivity and specificity
    results <- data.frame(matrix(nrow = 1, ncol = 4))
    colnames(results) <- c("alpha", "Sensitivity", "Specificity", "AUC")
    
    #a for loop to test alpha values incrementing by 0.01
    for (i in 1:99) {
      alpha = i/100
      
      #train the model with each respective alpha value
      model <- cv.glmnet(x = data.matrix(training[ , 2:ncol(training)]), y = as.factor(training$status), type.measure = "mse", alpha = alpha, family = "binomial", keep = TRUE)
      
      #use the model to classify the testing dataframe
      predict <- predict(object = model, s = model$lambda.min, newx = data.matrix(testing), type = "class") 
      
      #compute the sensitivity and specificity in a confusion matrix
      confusion_matrix <- confusionMatrix(data = as.factor(predict), reference = as.factor(validation$status))
      
      #change the predictions into an ordered factor
      predict <- factor(predict, ordered = TRUE)
      
      #calculated the AUC for model evaluation
      auc <- auc(validation$status, predict)
      
      #save the results for this alpha value in the results dataframe
      results[i, ] <- c(alpha, confusion_matrix$byClass["Sensitivity"] , confusion_matrix$byClass["Specificity"], auc) 
    }
    
    #plot the sensitivity and specificity for each alpha value to visually display the changes 
    print(ggplot(data = results) +
            geom_line(mapping = aes(x = alpha, y = Sensitivity, color = "red")) +
            geom_line(mapping = aes(x = alpha, y = Specificity, color = "blue")) +
            ggtitle("Alpha values vs sensitivity and Specificity") +
            ylab("value") +
            scale_color_manual(labels = c("Sensitivity", "Specificity"), values = c("red", "blue")) +
            guides(color = guide_legend("measure")))
    
    #filter the results for the alpha values that have the lowest difference between their sensitivity and specificity
    results <- results %>%
      mutate(difference = abs(Sensitivity - Specificity), .after = Specificity) %>%
      filter(difference == min(difference))
    
    #return the results dataframe
    return(results)
  }
}

  
#choose between "ridge", "lasso" and "elastic" to pick the regression
#save the results from the ridge regression classifier
final_results_df[1, ] <- elastic_net("ridge")

#save the results from the LASSO classifier
final_results_df[2, ] <- elastic_net("lasso")

#save the results from the elastic net classifier
elastic_regression <- elastic_net("elastic")

#take the first of the optimal alpha values from the elastic net classifier and save them in the final_results dataframe
final_results_df[3, ] <- c("elastic net", "alpha", elastic_regression[1, ])

#remove unneeded variables
rm(elastic_net)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# model: k nearest neighbour
# 
# Description: This section includes the code for the K nearest neighbour supervised machine learning classifier. It will compute Sensitivity, Specificity and 
# AUC. The best model will be filtered by the k value (the number of neighbours to consider) and this will be chosen as the k value that yields the lowest
# difference between the Sensitivity and Specificity
#
# Parameter: The parameter that will be changed for tuning in this model is the k value. The k will be incremented by 2 with only odd numbered k values being
# considered in order to avoid the instance of a draw 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#create a dataframe to contain the results of the classifier
knn_results <- as.data.frame(matrix(nrow = 1, ncol = 4))
colnames(knn_results) <- c("k", "Sensitivity", "Specificity", "AUC")

#set a counter to increment the dataframe
j = 0

#use a for loop to test every odd-numbered potential k value
for (i in seq(1, nrow(training), 2)) {
  #increment the dataframe counter
  j = j+1
  
  #train and test the knn model using i as the number of neighbours to consider
  knn <- knn(train = training[ , 2:ncol(training)], test = testing, cl = as.factor(training$status), k = i)
  
  #set the results of the test as an ordered factor
  knn <- factor(knn, ordered = TRUE)
  
  #create a confusion matrix to obtain the sensitivity and specificity
  confusion_matrix <- confusionMatrix(data = knn, reference = as.factor(validation$status))
  
  #calculate the AUC value for model analysis
  auc <- auc(validation$status, knn)
  
  #save the results of the model into the knn_results dataframe
  knn_results[j, ] <- c(i, confusion_matrix$byClass["Sensitivity"] , confusion_matrix$byClass["Specificity"], auc)
}

#plot the sensitivity and specificity at each k value to visually display the results
ggplot(data = knn_results) +
  geom_line(mapping = aes(x = k, y = Sensitivity, color = "red")) +
  geom_line(mapping = aes(x = k, y = Specificity, color = "blue")) +
  ylab("value") +
  scale_color_manual(labels = c("Sensitivity", "Specificity"), values = c("red", "blue")) +
  guides(color = guide_legend("measure")) +
  ggtitle("k values vs Sensitivity and Specificity")

#filter the results for the k values that yield the lowest difference in sensitivity and specificity
knn_results <- knn_results %>%
  mutate(difference = abs(Sensitivity - Specificity), .before = AUC) %>%
  filter(difference == min(difference))

#save the first instance of knn_results into the final_results dataframe
final_results_df[4, ] <- c("knn", "k", knn_results[1, ])

#remove unneeded variables
rm(j, i, knn, confusion_matrix)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# model: Classification and Regression trees (CART)
# 
# Description: This section includes the code for the CART supervised machine learning classifier. It will compute Sensitivity, Specificity and 
# AUC. The best model will be filtered by the number of terminal nodes (the number of nodes at the bottom of the tree) and this will be chosen as the number of # nodes that yield the lowest difference between the Sensitivity and Specificity
#
# Parameter: The parameter that will be changed for tuning in this model is the number of nodes. classification error rate will be used as the basis for 
# determining which number of nodes is the best (lowest classification error rate is the best) 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#generate a classification tree using the training data
class_tree <- tree(as.factor(status) ~ ., data = training)

#prune the tree and determine the best number of nodes using the error rate as the cost function
cv_tree <- cv.tree(class_tree, FUN = prune.misclass)

#save the number of nodes that resulted in the lowest error rate
best_size <- cv_tree$size[which(cv_tree$dev == min(cv_tree$dev))]

#create a CART dataframe to save all model performance results
CART <- as.data.frame(matrix(nrow = length(best_size), ncol = 4))
colnames(CART) <- c("nodes", "Sensitivity", "Specificity", "AUC")

#model performance parameters are conducted for each node count that was found to be the best
for (i in 1:length(best_size)) {
  #the tree is pruned using the number of nodes determined in best_size
  prune_tree <- prune.misclass(class_tree, best = best_size[i])
  
  #the pruned tree is used to classify the testing dataset
  class_pred <- predict(prune_tree, testing, type = "class")
  
  #a confusion matrix is generated to determine the Sensitivity and specificity of the model
  confusion_matrix <- confusionMatrix(as.factor(class_pred), as.factor(validation$status))
  
  #convert the classification into an ordered factor
  class_pred <- factor(class_pred, ordered = TRUE)
  
  #calculate the AUC 
  auc <- auc(validation$status, class_pred)
  
  #save the model performance parameters
  CART[i, ] <- c(best_size[i], confusion_matrix$byClass["Sensitivity"] , confusion_matrix$byClass["Specificity"], auc)
}

#Filter the CART dataframe and retain only the rows that have the lowest difference between sensitivity and specificity
CART <- CART %>%
  mutate(difference = abs(Sensitivity - Specificity), .before = AUC) %>%
  filter(difference == min(difference))

#save the first row in the CART dataframe to the final results dataframe
final_results_df[5, ] <- c("CART", "nodes", CART[1, ])

#remove unneeded variables
rm(prune_tree, class_pred, confusion_matrix, best_size, class_tree, cv_tree, i, auc)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# model: support vector machine (SVM)
# 
# Description: This section includes the code for the SVM supervised machine learning classifier. It will compute Sensitivity, Specificity and 
# AUC. The best model will be filtered by the cost function (the degree to which the classifier is penalized for an incorrect classification) and this will be 
# chosen as the cost that most reduces classification error. The support vector machine will run using radial kernels
# and linear kernels and the two will be compared. The radial kernel will additionally be chosen based on the best gamma value
#
# Parameter: The parameter that will be changed for tuning in this model is the cost function. 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#create a dataframe to hold the results from both svm classifiers (linear and radial)
svm_df <- as.data.frame(matrix(nrow = 1, ncol = 5))
colnames(svm_df) <- c("type", "cost", "Sensitivity", "Specificity", "AUC")

#compute the model parameters for both linear and radial kernels
for (i in seq(1:2)) {
  
  #in the first instance of the loop, linear kernels will be used
  if (i == 1) {
    type <- "linear"
    
    #tune the model using linear kernels
    svm_tuned <- tune(svm, as.factor(status) ~ ., data = training, kernel = type, ranges = list(cost = c(seq(1,100,1))))
    color <- "steelblue"
  }
  
  #in the second instance of the loop, radial kernels will be used
  else {
    type <- "radial"
    
    #tune the model using radial kernels
    svm_tuned <- tune(svm, as.factor(status) ~ ., data = training, kernel = "radial", ranges = list(cost = c(seq(1, 10, 1)), gamma = c(seq(0,1,0.1))))
    color <- "#A64543"
  }
  
  #use the tuned model to classify the testing dataset
  pred <- predict(svm_tuned$best.model, testing)
  
  #calculate the sensitivity and specificity for the model
  confusion_matrix <- confusionMatrix(pred, as.factor(validation$status))
  
  #change the model classifier into an ordered factor
  pred <- factor(pred, ordered = TRUE)
  
  #calculate the ROC for the model
  roc <- roc(predictor = pred, response = as.factor(validation$status))
  
  #print an ROC curve to visually display the model results
  print(ggroc(roc, colour = color, size = 2) +
    ggtitle(paste0("ROC for ", type, " svm")) +
    geom_abline(slope = 1, intercept = 1))
  
  #calculate the AUC for the model
  auc <- auc(validation$status, pred)
  
  #save the results into the svm dataframe
  svm_df[i, ] <- c(type, svm_tuned$best.parameters[1,1], confusion_matrix$byClass["Sensitivity"] , confusion_matrix$byClass["Specificity"], auc) 
}

#add a column showing the difference between the sensitivity and the specificity. 
svm_df <- svm_df %>%
  mutate(difference = abs(as.numeric(Sensitivity) - as.numeric(Specificity)), .before = AUC)

#save the results of each model into the final_results dataframe
final_results_df[6, ] <- c(paste0(svm_df[1,1], " svm"), "cost", svm_df[1 , 2:6])
final_results_df[7, ] <- c(paste0(svm_df[2,1], " svm"), "cost", svm_df[2 , 2:6])

#remove unneeded variables
rm(pred, confusion_matrix, roc, auc, i, type, svm_tuned, color) 

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# model: bagged classification and regression trees (bagged CART)
# 
# Description: This section includes the code for the bagged CART supervised machine learning classifier. It will compute Sensitivity, Specificity and 
# AUC. This model will use the random forest function but will set the value of mtry to be equal to the number of predictors which will allow the random forest
# classifier to act as a bagged classification and regression tree
#
# Parameter: The parameter that will be set for this model is mtry = number of predictors
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#use the random forest classifier with mtry = number of predictors to make a bagged CART classifier
CART_bagging <- randomForest(as.factor(status) ~ ., data = training, mtry = ncol(training)-1, ntree = 1000)

#use the classifier to classify the testing dataset
CART_pred <- predict(CART_bagging, testing, type = "class")

#create a confusion matrix to calculate the sensitivity and specificity
confusion_matrix <- confusionMatrix(as.factor(CART_pred), as.factor(validation$status))

#change the classifier output to an ordered factor 
CART_pred <- factor(CART_pred, ordered = TRUE)

#calculate the AUC for the model
auc <- auc(validation$status, CART_pred)

#calculate the difference between the sensitivity and the specificity
difference <- abs(confusion_matrix$byClass["Sensitivity"] - confusion_matrix$byClass["Specificity"])

#calculate the ROC to plot the results
roc_CART <- roc(predictor = CART_pred, response = as.factor(validation$status))

#print an ROC curve to visually display the classifier results
ggroc(roc_CART, colour = "steelblue", size = 2) +
        ggtitle("ROC for bagged CART") +
        geom_abline(slope = 1, intercept = 1)

#save the classifiers parameters to the final results dataframe
final_results_df[8, ] <- c("Bagged CART", "variables sampled", ncol(training)-1, confusion_matrix$byClass["Sensitivity"] , confusion_matrix$byClass["Specificity"], difference, auc)

#remove unneeded variables
rm(CART_bagging, CART_pred, confusion_matrix, auc, roc_CART, difference)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# model: Random forest
# 
# Description: This section includes the code for the random forest supervised machine learning classifier. It will compute Sensitivity, Specificity and 
# AUC. This model be trained using repeated cross validation with a random search. This means a random number of variables (mtry) will be selected to be  
# subsampled at split and the optimal mtry will be selected based on accuracy 
#
# Parameter: The parameter that will be tuned for optimization is the number of variables that will be selected to be subsampled (mtry)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#tune the parameters of the training algorithm to use repeated cross validation and random searching
control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3,
                        search = 'random')

#train the model with 15 different values for mtry and determine which results in the best accuracy
tune_rf <- train(status ~., data = training, method = "rf", metric = "Accuracy", tuneLength = 10, trControl = control, ntree = 1000)

#plot the results of the tuning and show which has the best accuracy
plot(tune_rf, main = "Accuracy of different mtry values for random forest")

#extract the best model from the training
ranfor <- tune_rf$finalModel

#classify the samples in the testing dataset
ranfor_pred <- predict(ranfor, testing)

#use a confusion matrix to obtain the specificity and sensitivity of the model
confusion_matrix <- confusionMatrix(as.factor(ranfor_pred), as.factor(validation$status))

#convert the classified output into an ordered factor
ranfor_pred <- factor(ranfor_pred, ordered = TRUE)

#calculate the AUC for the model
auc <- auc(validation$status, ranfor_pred)

#calculate the difference between the sensitivity and the specificity
difference <- abs(confusion_matrix$byClass["Sensitivity"] - confusion_matrix$byClass["Specificity"])

#save the results to the final_results dataframe
final_results_df[9, ] <- c("random forest", "variables sampled", tune_rf$bestTune, confusion_matrix$byClass["Sensitivity"] , confusion_matrix$byClass["Specificity"], difference, auc)

#remove unneeded variables
rm(tune_rf, ranfor, ranfor_pred, confusion_matrix, auc, control, difference, testing, training, validation)

#round all numbers in the final resuslts dataframe to 4 digits
final_results_df[ ,4] <- round(as.numeric(final_results_df[ , 4]), 4)
final_results_df[ ,5] <- round(as.numeric(final_results_df[ , 5]), 4)
final_results_df[ ,6] <- round(as.numeric(final_results_df[ , 6]), 4)
final_results_df[ ,7] <- round(as.numeric(final_results_df[ , 7]), 4)

final_results_df