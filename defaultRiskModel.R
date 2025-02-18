# Loading the Libraries we may need
library(tidyverse)
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(Metrics)
library(rpart)
library(DescTools)
library(klaR)
library(modelr)
library(pROC)
library(ROSE)

# Loading in data and removing the unneeded ID variable
Data <- read.csv("CreditDefault.csv")
Data <- Data %>% dplyr::select(-ID)

# Setting the random seed for reproducibility 
# and creating our training/test split
set.seed(2112)
indx <- sample(nrow(Data), nrow(Data) * 0.80)
Train_Data <- Data[indx, ]
Test_Data <- Data[-indx, ]

# Training a basic logistic regression GLM model
GLM_Model <- train(default.payment.next.month ~ ., data = Train_Data,
                   method = "glm", family = binomial)

# Making Predictions
GLM_Pred <- predict(GLM_Model, Test_Data, type = "raw")
GLM_Pred <- ifelse(GLM_Pred >= 0.5, 1, 0)
Obs <- Test_Data$default.payment.next.month

# Making the confusion matrix for analysis
GLM_CM <- confusionMatrix(as.factor(GLM_Pred), as.factor(Obs), positive = "1")
print(GLM_CM)

# Making the ROC curve and AUC score for analysis
GLM_ROC <- roc(GLM_Pred, Obs)
plot(GLM_ROC, main = "GLM_ROC", col = "red")
auc(GLM_ROC)

# Checking for imbalance and getting minority percentage.
maj_class <- sum(Train_Data$default.payment.next.month == 0)
print(maj_class)
min_class <- sum(Train_Data$default.payment.next.month == 1)
print(min_class)
imbal <- min_class/(min_class+maj_class)
print(imbal)

#Setting train control for 5 fold cross validation
TrCtrl <- trainControl(method = "cv", number = 5)

# Training random forest model.
RF_Model <- train(default.payment.next.month ~ ., data = Train_Data,
                  method = "rf", trControl = TrCtrl,
                  tuneGrid = data.frame(.mtry = 5))

print(RF_Model)

# Making predictions
RF_Pred <- predict(RF_Model, Test_Data, type = "raw")
RF_Pred <- ifelse(RF_Pred >= 0.5, 1, 0)

# Making the confusion matrix for analysis
RF_CM <- confusionMatrix(as.factor(RF_Pred), as.factor(Obs), positive = "1")
print(RF_CM)

# Making the ROC Curve and getting AUC score for analysis
RF_ROC <- roc(RF_Pred, Obs)
plot(RF_ROC, main ="RF_ROC", col = "red")
auc(RF_ROC)

# Defining class weights
CW <- ifelse(Train_Data$default.payment.next.month == 0, 1, 10)

# Creating a random forest model with class weights
CS_Model <- train(default.payment.next.month ~ ., data = Train_Data,
                  method = "rf", trControl = TrCtrl,
                  tuneGrid = data.frame(.mtry = 5),
                  weights = CW)
print(CS_Model)

# Making predictions
CS_Pred <- predict(CS_Model, Test_Data, type = "raw")
CS_Pred <- ifelse(CS_Pred >= 0.5, 1, 0)

# Making the confusion matrix for analysis
CS_CM <- confusionMatrix(as.factor(CS_Pred), as.factor(Obs), positive = "1")
print(CS_CM)

# Making the ROC Curve and getting AUC score for analysis
CS_ROC <- roc(CS_Pred, Obs)
plot(CS_ROC, main ="CS_ROC", col = "red")
auc(CS_ROC)

# Testing for what variables may not be important
test_imp <- varImp(CS_Model)
plot(test_imp, main = "Variable Importance for CS_Model")

# Removing Variables of low importance
Train_Data <- Train_Data %>% 
  dplyr::select(-SEX, -MARRIAGE, -EDUCATION, -PAY_6, -PAY_4,
                -PAY_5, -PAY_3)
Test_Data <- Test_Data %>% 
  dplyr::select(-SEX, -MARRIAGE, -EDUCATION, -PAY_6, -PAY_4,
                -PAY_5, -PAY_3)


# Training a random forest model specifically for recall
RC_Model <- train(default.payment.next.month ~ ., data = Train_Data, 
                  method = "rf", trControl = TrCtrl, metric = "recall",
                  tuneGrid = data.frame(.mtry = 5))
print(RC_Model)

# Making Predictions
RC_Pred <- predict(RC_Model, Test_Data, type ="raw")
RC_Pred <- ifelse(RC_Pred >= 0.5, 1, 0)

# Making the confusion matrix
RC_CM <- confusionMatrix(as.factor(RC_Pred), as.factor(Obs), positive = "1")
print(RC_CM)

# Making the ROC Curve and getting AUC score for analysis
RC_ROC <- roc(RC_Pred, Obs)
plot(RC_ROC, main = "RC_ROC", col = "red")
auc(RC_ROC)

# upsampling our training and test data
TRAIN_R = ROSE(default.payment.next.month ~ .,
               data = Train_Data, seed = 2112)$data

TEST_R = ROSE(default.payment.next.month ~ .,
              data = Test_Data, seed = 2112)$data

# Training a random forest model with our upsampled data from ROSE 
ROSE_Model <- train(default.payment.next.month ~ ., data = TRAIN_R,
                  method = "rf", trControl = TrCtrl,
                  tuneGrid = data.frame(.mtry = 5))
print(ROSE_Model)

# Making predictions
ROSE_Pred <- predict(ROSE_Model, TEST_R, type = "raw")
ROSE_Pred <- ifelse(ROSE_Pred >= 0.5, 1, 0)

# Making the confusion matrix for analysis
ROSE_CM <- confusionMatrix(as.factor(ROSE_Pred),
                           as.factor(TEST_R$default.payment.next.month),
                           positive = "1")
print(ROSE_CM)

# Making the ROC Curve and getting AUC score for analysis
ROSE_ROC <- roc(ROSE_Pred, TEST_R$default.payment.next.month)
plot(ROSE_ROC, main = "ROSE_ROC at 0.5 Threshold", col = "red")
auc(ROSE_ROC)

# Analysis of model at 0.4 threshold

# Making predictions
ROSE_Pred <- predict(ROSE_Model, TEST_R, type = "raw")
ROSE_Pred <- ifelse(ROSE_Pred >= 0.4, 1, 0)
R_Obs <- TEST_R$default.payment.next.month

# Making the confusion matrix for analysis
ROSE_CM <- confusionMatrix(as.factor(ROSE_Pred), as.factor(R_Obs),
                           positive = "1")
print(ROSE_CM)

# Making the ROC Curve and getting AUC score for analysis
ROSE_ROC <- roc(ROSE_Pred, R_Obs)
plot(ROSE_ROC, main = "ROSE_ROC at 0.4 Threshold", col = "red")
auc(ROSE_ROC)

# Finalizing the model
Training_Data <- TRAIN_R
Testing_Data <- TEST_R

Model <- train(default.payment.next.month ~ ., data = Training_Data,
                    method = "rf", trControl = TrCtrl,
                    tuneGrid = data.frame(.mtry = 5))
print(Model)

Predictions <- predict(Model, Testing_Data, type = "raw")
Predictions <- ifelse(Predictions >= 0.4, 1, 0)

Observations <- Testing_Data$default.payment.next.month
Confusion_Matrix <- confusionMatrix(as.factor(Predictions),
                                    as.factor(Observations), positive = "1")
print(Confusion_Matrix)

ROC_Curve <- roc(Predictions, Observations)
plot(ROC_Curve, main = "ROC_Curve", col = "red")
auc(ROC_Curve)
