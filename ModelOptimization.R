library(readxl)
library(caret)
library(car)
library(yardstick)
library(class)
library(randomForest)
library(ggplot2)
library(pROC)
library(kernlab)
library(e1071)
library(gplots)
library(xgboost)
library(corrplot)
library(cvms)
library(tibble)
library(nnet)


# Read and process data
data <- read_excel("data.xlsx", sheet="Sheet1")

names(data)[names(data) == "VORP per season in NBA"] <- "VORP_per_season_in_NBA"
names(data)[names(data) == "FG%"] <- "FG_percent"
names(data)[names(data) == "3P%"] <- "Three_percent"
names(data)[names(data) == "FT%"] <- "FT_percent"
names(data)[names(data) == "Years in college"] <- "years_in_college"
names(data)[names(data) == "Mock draft"] <- "mock_draft"
names(data)[names(data) == "GS%"] <- "GS_percent"
names(data)[names(data) == "draft rank"] <- "draft_rank"
data$Tier <- factor(data$Tier, levels = c("Star","Above average","Bench player","Reserve"),
                    labels = c("Star","Above average","Bench player","Reserve"))

# Split data into train and test
train_data <- data[!data$`draft year` %in% c(2015, 2016, 2017), ]
test_data <- data[data$`draft year` %in% c(2015, 2016, 2017), ]



### MODEL OPTIMIZATION

class_weights <- c(1/(as.numeric(table(train_data$Tier)[1])/nrow(train_data)),
                   1/(as.numeric(table(train_data$Tier)[2])/nrow(train_data)),
                   1/(as.numeric(table(train_data$Tier)[3])/nrow(train_data)),
                   1/(as.numeric(table(train_data$Tier)[4])/nrow(train_data)))
control <- trainControl(method = "repeatedcv",
                        number = 5,    # Number of CV folds
                        repeats = 3,   # Number of repeated CV
                        search = "grid")  # Use grid search for hyperparameter tuning



### SVM MODEL OPTIMIZATION



### RADIAL

# Define the range of hyperparameters to search
cost_values <- 2^(-4:5)      # C (cost) values
sigma_values <- c(0.071, 0.224, 0.707, 2.236, 7.071, 22.361) # Sigma values for RBF kernel (corresponds to gamma (0.001, 0.01, 0.1, 1, 10, 100))

# Create a data frame with all combinations of hyperparameters
svmGrid <- expand.grid(C = cost_values,
                       sigma = sigma_values)
model_weights <- ifelse(train_data$Tier == "Star", class_weights[1],
                        ifelse(train_data$Tier == "Above average", class_weights[2],
                               ifelse(train_data$Tier == "Bench player", class_weights[3],class_weights[4])))

set.seed(123)
svmModelRadial <- train(Tier ~ MP + FG_percent + Three_percent + FT_percent + TRB + AST + STL + BLK + TOV 
                        + PF + SOS + years_in_college + mock_draft + PTS_per_40,
                        data = train_data,
                        method = "svmRadial",
                        weights = model_weights,
                        metric = "Accuracy",
                        trControl = control,
                        preProcess = c("center", "scale"), # Preprocess the data (optional)
                        tuneGrid = svmGrid)


### LINEAR
cost_values <- 2^(-4:5)      # C (cost) values
svmGrid <- expand.grid(C = cost_values)

set.seed(123)
svmModelLinear <- train(Tier ~ MP + FG_percent + Three_percent + FT_percent + TRB + AST + STL + BLK + TOV 
                        + PF + SOS + years_in_college + mock_draft + PTS_per_40,
                        data = train_data,
                        method = "svmLinear",
                        weights = model_weights,
                        metric = "Accuracy",
                        trControl = control,
                        preProcess = c("center", "scale"), # Preprocess the data (optional)
                        tuneGrid = svmGrid)


### POLYNOMIAL

cost_values <- 2^(-4:5)      # C (cost) values
degree_values <- 1:4         # Degree values for polynomial kernel
scale_values <- seq(0.1, 1, 0.1)  # Scale values for polynomial kernel

svmGrid <- expand.grid(C = cost_values,
                       degree = degree_values,
                       scale = scale_values)

set.seed(123)
svmModelPolynomial <- train(Tier ~ MP + FG_percent + Three_percent + FT_percent + TRB + AST + STL + BLK + TOV 
                            + PF + SOS + years_in_college + mock_draft + PTS_per_40,
                            data = train_data,
                            method = "svmPoly",
                            weights = model_weights,
                            metric = "Accuracy",
                            trControl = control,
                            preProcess = c("center", "scale"), # Preprocess the data (optional)
                            tuneGrid = svmGrid)


cat(" Linear Kernel Accuracy:", svmModelLinear$results[row.names(svmModelLinear$bestTune),]$Accuracy, "\n", 
    "Polynomial Kernel Accuracy:", svmModelPolynomial$results[row.names(svmModelPolynomial$bestTune),]$Accuracy, "\n",
    "Radial Kernel Accuracy:", svmModelRadial$results[row.names(svmModelRadial$bestTune),]$Accuracy, sep = " ")


# TRAIN SVM ON ENTIRE TRAIN DATASET
class_weights_svm <- class_weights
names(class_weights_svm) <- levels(train_data$Tier)
svm_train_model <- svm(Tier ~ MP + FG_percent + Three_percent + FT_percent + TRB + AST + STL + BLK + TOV 
                       + SOS + years_in_college + mock_draft + PTS_per_40, data = train_data,
                       kernel = "linear", cost = svmModelLinear$bestTune$C, scale = TRUE,
                       class.weights = class_weights_svm)#gamma = 1/(2*svmModelRadial$bestTune$sigma^2),
train_svm_predictions <- factor(svm_train_model$fitted)

conf_mat <- confusionMatrix(train_svm_predictions, train_data$Tier)
# PLOT CONFUSION MATRIX
cfm <- as_tibble(conf_mat$table)
plot_confusion_matrix(cfm, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n",
                      class_order = c("Reserve","Bench player","Above average","Star"))
# GET TRAIN ACCURACY FOR SVM MODEL
accuracy_train_svm <- mean(train_svm_predictions == train_data$Tier)
weighted_accuracy_train_svm <- sum(diag(conf_mat$table) * class_weights) / sum(t(conf_mat$table) * class_weights)

# TEST MODEL
test_svm_predictions <- predict(svm_train_model, test_data[, -c(19,20)])
conf_mat <- confusionMatrix(test_svm_predictions, test_data$Tier)
# PLOT CONFUSION MATRIX
cfm <- as_tibble(conf_mat$table)
plot_confusion_matrix(cfm,
                      target_col = "Reference",
                      prediction_col = "Prediction",
                      counts_col = "n",
                      class_order = c("Reserve","Bench player","Above average","Star"))
# GET TEST ACCURACY FOR SVM MODEL
accuracy_test_svm <- mean(test_svm_predictions == test_data$Tier)
weighted_accuracy_test_svm <- sum(diag(conf_mat$table) * class_weights) / sum(t(conf_mat$table) * class_weights)






### RANDOM FOREST OPTIMIZATION

# Define the range of hyperparameters to search
ntree_values <- c(100, 250, 500)
mtry_values <- c(2, 4, 6, 8, 10, 12, 14) # Number of variables tried at each split

# Create a data frame with all combinations of hyperparameters
rfGrid <- expand.grid(mtry = mtry_values)

column_names <- c("ntree_value", "mtry", "accuracy")
rfModelsResults <- data.frame(matrix(ncol = length(column_names), nrow = 0))
for (ntree_value in ntree_values) {
  set.seed(123)
  rfModel <- train(Tier ~ MP + FG_percent + Three_percent + FT_percent + TRB + AST + STL + BLK + TOV 
                   + PF + SOS + years_in_college + mock_draft + PTS_per_40,
                   data = train_data,
                   method = "rf",
                   weights = model_weights,
                   metric = "Accuracy",
                   trControl = control,
                   tuneGrid = rfGrid,
                   ntree = ntree_value)
  new_row <- data.frame(col1 = ntree_value, col2 =  rfModel$bestTune$mtry, col3 = rfModel$results[row.names(rfModel$bestTune),]$Accuracy)
  rfModelsResults <- rbind(rfModelsResults, new_row)
}
colnames(rfModelsResults) <- column_names


# TRAIN RF ON ENTIRE TRAIN DATASET

rf_train_model <- randomForest(Tier ~ MP + FG_percent + Three_percent + FT_percent + TRB + AST + STL + BLK + TOV 
                               + SOS + years_in_college + mock_draft + PTS_per_40, data = train_data,
                               ntree = rfModelsResults[which.max(rfModelsResults$accuracy), ]$ntree_value,
                               mtry = rfModelsResults[which.max(rfModelsResults$accuracy), ]$mtry,
                               classwt = class_weights)
train_rf_predictions <- factor(rf_train_model$predicted)
conf_mat <- confusionMatrix(train_rf_predictions, train_data$Tier)
# PLOT CONFUSION MATRIX
cfm <- as_tibble(conf_mat$table)
plot_confusion_matrix(cfm, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n",
                      class_order = c("Reserve","Bench player","Above average","Star"))
# GET TRAIN ACCURACY FOR RF MODEL
accuracy_train_rf <- mean(train_rf_predictions == train_data$Tier)
weighted_accuracy_train_rf <- sum(diag(conf_mat$table) * class_weights) / sum(t(conf_mat$table) * class_weights)

# TEST MODEL
test_rf_predictions <- predict(rf_train_model, test_data[, -c(19,20)])
conf_mat <- confusionMatrix(test_rf_predictions, test_data$Tier)
cfm <- as_tibble(conf_mat$table)
plot_confusion_matrix(cfm, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n",
                      class_order = c("Reserve","Bench player","Above average","Star"))
# GET TEST ACCURACY FOR RF MODEL
accuracy_test_rf <- mean(test_rf_predictions == test_data$Tier)
weighted_accuracy_test_rf <- sum(diag(conf_mat$table) * class_weights) / sum(t(conf_mat$table) * class_weights)

# PLOT FEATURE IMPORTANCE
importance_matrix <- importance(rf_train_model)
feature_importance <- data.frame(
  Feature = rownames(importance_matrix),
  Importance = importance_matrix[, "MeanDecreaseGini"]
)
ggplot(feature_importance, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(x = "Feature", y = "Importance (Mean Decrease Gini)", title = "Random Forest Feature Importance")






### XGBOOST OPTIMIZATION

data$Tier_numeric <- as.numeric(data$Tier) - 1

train_data <- data[!data$`draft year` %in% c(2015, 2016, 2017), ]
test_data <- data[data$`draft year` %in% c(2015, 2016, 2017), ]
train_data_xgb <- train_data[, c(6, 9, 15, 18, 21, 22, 23, 24, 25, 28, 32, 33, 37, 38)]
test_data_xgb <- test_data[, c(6, 9, 15, 18, 21, 22, 23, 24, 25, 28, 32, 33, 37, 38)]


xgbGrid <- expand.grid(nrounds = c(100, 250),
                       max_depth = c(3, 6, 10),
                       eta = c(0.01, 0.1, 0.2, 0.3),
                       gamma = c(0, 0.5, 1, 5),
                       colsample_bytree = c(0.5, 0.7, 0.9),
                       min_child_weight = c(1, 5),
                       subsample = c(0.5, 0.6, 0.7, 0.8))

set.seed(123)
xgbModel <- train(x = data.matrix(train_data_xgb[, -15]),
                  y = train_data$Tier,
                  method = "xgbTree",
                  weights = model_weights,
                  trControl = control,
                  tuneGrid = xgbGrid,
                  metric = "Accuracy",
                  nthread = 1)

# TRAIN XGBoost ON ENTIRE TRAIN DATASET
instance_weights <- class_weights[train_data$Tier]

train_matrix <- xgb.DMatrix(data.matrix(train_data_xgb[, -14]), label = as.numeric(train_data_xgb$Tier_numeric), weight = instance_weights)
test_matrix <- xgb.DMatrix(data.matrix(test_data_xgb[, -14]), label = as.numeric(test_data_xgb$Tier_numeric))

# Set the parameters for the XGBoost model
params <- list(
  objective = "multi:softmax",
  num_class = 4,
  booster = "gbtree",
  eta = xgbModel$bestTune$eta,
  gamma = xgbModel$bestTune$gamma,
  max_depth = 6,
  min_child_weight = xgbModel$bestTune$min_child_weight,
  subsample = xgbModel$bestTune$subsample,
  colsample_bytree = xgbModel$bestTune$colsample_bytree,
  eval_metric = "merror"
)

# Train the XGBoost model
set.seed(42)
xgb_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = xgbModel$bestTune$nrounds,
  watchlist = list(train = train_matrix, test = test_matrix),
  print_every_n = 10,
  early_stopping_rounds = 10
)

train_xgb_predictions <- predict(xgb_model, train_matrix)

convert_numeric_to_tiers <- function(values) {
  tiers <- c()
  for (pred in values) {
    if (pred == 0) {
      tiers <- append(tiers, "Star")
    }
    else if (pred == 1) {
      tiers <- append(tiers, "Above average")
    }
    else if (pred == 2) {
      tiers <- append(tiers, "Bench player")
    }
    else {
      tiers <- append(tiers, "Reserve")
    }
  }
  return(tiers)
}


conf_mat <- confusionMatrix(factor(convert_numeric_to_tiers(train_xgb_predictions)), train_data$Tier)

# PLOT CONFUSION MATRIX
cfm <- as_tibble(conf_mat$table)
plot_confusion_matrix(cfm, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n",
                      class_order = c("Reserve","Bench player","Above average","Star"))
# GET TRAIN ACCURACY FOR XGBOOST MODEL
accuracy_train_xgb <- mean(convert_numeric_to_tiers(train_xgb_predictions) == train_data$Tier)
weighted_accuracy_train_xgb <- sum(diag(conf_mat$table) * class_weights) / sum(t(conf_mat$table) * class_weights)


# TEST XGBOOST MODEL
#test_xgb_predictions <- predict(xgb_model, test_matrix)
conf_mat <- confusionMatrix(factor(convert_numeric_to_tiers(test_xgb_predictions)), test_data$Tier)
cfm <- as_tibble(conf_mat$table)
plot_confusion_matrix(cfm,
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n",
                      class_order = c("Reserve","Bench player","Above average","Star"))
# GET TEST ACCURACY FOR XGBOOST MODEL
accuracy_test_xgb <- mean(convert_numeric_to_tiers(test_xgb_predictions) == test_data$Tier)
weighted_accuracy_test_xgb <- sum(diag(conf_mat$table) * class_weights) / sum(t(conf_mat$table) * class_weights)

# PLOT FEATURE IMPORTANCE
importance_matrix <- xgb.importance(feature_names = colnames(train_data_xgb[, -14]), model = xgb_model)

# Create data frame for plotting
feature_importance <- data.frame(
  Feature = importance_matrix$Feature,
  Importance = importance_matrix$Gain
)

# Plot feature importance using ggplot2
ggplot(feature_importance, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(x = "Feature", y = "Importance (Gain)", title = "XGBoost Feature Importance")





### KNN OPTIMIZATION
train_data_knn <- train_data[, c(6, 9, 15, 18, 21, 22, 23, 24, 25, 28, 32, 33, 37)]
test_data_knn <- test_data[, c(6, 9, 15, 18, 21, 22, 23, 24, 25, 28, 32, 33, 37)]
train_labels <- train_data$Tier
test_labels <- test_data$Tier

# Define the hyperparameters
k_values <- c(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39)
tune_grid <- expand.grid(k = k_values)

# Train the KNN model
set.seed(123)
model <- train(
  x = train_data_knn,
  y = train_labels,
  method = "knn",
  weights = model_weights,
  metric = "Accuracy",
  tuneGrid = tune_grid,
  trControl = control,
  preProc = c("center", "scale")
)

plot(model, main="Accuracy for each k", xlab="K-Neighbors")


# TRAIN KNN MODEL ON ENTIRE TRAIN DATASET

# Normalize the data
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
train_data_normalized <- as.data.frame(lapply(train_data_knn, normalize))
test_data_normalized <- as.data.frame(lapply(test_data_knn, normalize))

k <- 17
train_predictions <- knn(train_data_normalized, train_data_normalized, train_labels, k = k)

conf_mat <- confusionMatrix(train_predictions, train_data$Tier)

# PLOT CONFUSION MATRIX
cfm <- as_tibble(conf_mat$table)
plot_confusion_matrix(cfm, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n",
                      class_order = c("Reserve","Bench player","Above average","Star"))
# GET TRAIN ACCURACY FOR KNN MODEL
accuracy_train_knn <- mean(train_predictions == train_data$Tier)
weighted_accuracy_train_knn <- sum(diag(conf_mat$table) * class_weights) / sum(t(conf_mat$table) * class_weights)

# TEST KNN MODEL
test_predictions <- knn(train_data_normalized, test_data_normalized, train_labels, k = k)
conf_mat <- confusionMatrix(test_predictions, test_data$Tier)

# PLOT CONFUSION MATRIX
cfm <- as_tibble(conf_mat$table)
plot_confusion_matrix(cfm, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n",
                      class_order = c("Reserve","Bench player","Above average","Star"))
# GET TEST ACCURACY FOR KNN MODEL
accuracy_test_knn <- mean(test_predictions == test_data$Tier)
weighted_accuracy_test_knn <- sum(diag(conf_mat$table) * class_weights) / sum(t(conf_mat$table) * class_weights)





# MULTINOMIAL LOGISTIC REGRESSION
train_lr <- multinom(Tier ~ MP + FG_percent + Three_percent + FT_percent + TRB + AST + STL + BLK + TOV 
                     + SOS + years_in_college + mock_draft + PTS_per_40, data = train_data, weights = model_weights)
train_lr_predictions <- factor(predict(train_lr, train_data))


conf_mat <- confusionMatrix(train_lr_predictions, train_data$Tier)
# PLOT CONFUSION MATRIX
cfm <- as_tibble(conf_mat$table)
plot_confusion_matrix(cfm, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n",
                      class_order = c("Reserve","Bench player","Above average","Star"))

# CALCULATE ACCURACY
accuracy_train_lr <- mean(train_lr_predictions == train_data$Tier)
weighted_accuracy_train_lr <- sum(diag(conf_mat$table) * class_weights) / sum(t(conf_mat$table) * class_weights)


# TEST MODEL
test_lr_predictions <- factor(predict(train_lr, test_data))
conf_mat <- confusionMatrix(test_lr_predictions, test_data$Tier)
# PLOT CONFUSION MATRIX
cfm <- as_tibble(conf_mat$table)
plot_confusion_matrix(cfm, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n",
                      class_order = c("Reserve","Bench player","Above average","Star"))
accuracy_test_lr <- mean(test_lr_predictions == test_data$Tier)
weighted_accuracy_test_lr <- sum(diag(conf_mat$table) * class_weights) / sum(t(conf_mat$table) * class_weights)





### BENCHMARK MODEL

data$Tier_numeric <- as.numeric(data$Tier) - 1

train_data <- data[!data$`draft year` %in% c(2015, 2016, 2017), ]
test_data <- data[data$`draft year` %in% c(2015, 2016, 2017), ]
train_data_xgb <- train_data[, c(6, 9, 15, 18, 21, 22, 23, 24, 25, 28, 32, 33, 37, 38)]
test_data_xgb <- test_data[, c(6, 9, 15, 18, 21, 22, 23, 24, 25, 28, 32, 33, 37, 38)]
train_data_xgb <- train_data[, c(34, 38)]
test_data_xgb <- test_data[, c(34, 38)]


xgbGrid <- expand.grid(nrounds = c(100, 250),
                       max_depth = c(3, 6, 10),
                       eta = c(0.01, 0.1, 0.2, 0.3),
                       gamma = c(0, 0.5, 1, 5),
                       colsample_bytree = c(0.5, 0.7, 0.9),
                       min_child_weight = c(1, 5),
                       subsample = c(0.5, 0.6, 0.7, 0.8))

set.seed(123)
xgbModel <- train(x = data.matrix(train_data_xgb[, -15]),
                  y = train_data$Tier,
                  method = "xgbTree",
                  weights = model_weights,
                  trControl = control,
                  tuneGrid = xgbGrid,
                  metric = "Accuracy",
                  nthread = 1)

# TRAIN BENCHMARK XGBoost
instance_weights <- class_weights[train_data$Tier]

train_matrix <- xgb.DMatrix(data.matrix(train_data_xgb[, -2]), label = as.numeric(train_data_xgb$Tier_numeric), weight = instance_weights)
test_matrix <- xgb.DMatrix(data.matrix(test_data_xgb[, -2]), label = as.numeric(test_data_xgb$Tier_numeric))

# Set the parameters for the BENCHMARK XGBoost model
params <- list(
  objective = "multi:softmax",
  num_class = 4,
  booster = "gbtree",
  eta = xgbModel$bestTune$eta,
  gamma = xgbModel$bestTune$gamma,
  max_depth = 6,
  min_child_weight = xgbModel$bestTune$min_child_weight,
  subsample = xgbModel$bestTune$subsample,
  colsample_bytree = xgbModel$bestTune$colsample_bytree,
  eval_metric = "merror"
)

# Train the BENCHMARK XGBoost model
set.seed(42)
xgb_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = xgbModel$bestTune$nrounds,
  watchlist = list(train = train_matrix, test = test_matrix),
  print_every_n = 10,
  early_stopping_rounds = 10
)

train_xgb_predictions <- predict(xgb_model, train_matrix)


conf_mat <- confusionMatrix(factor(convert_numeric_to_tiers(train_xgb_predictions)), train_data$Tier)

# PLOT CONFUSION MATRIX
cfm <- as_tibble(conf_mat$table)
plot_confusion_matrix(cfm, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n",
                      class_order = c("Reserve","Bench player","Above average","Star"))
# GET TRAIN ACCURACY FOR BENCHMARK XGBOOST MODEL
accuracy_train_xgb <- mean(convert_numeric_to_tiers(train_xgb_predictions) == train_data$Tier)
weighted_accuracy_train_xgb <- sum(diag(conf_mat$table) * class_weights) / sum(t(conf_mat$table) * class_weights)


# TEST BENCHMARK XGBOOST MODEL
test_xgb_predictions <- predict(xgb_model, test_matrix)
conf_mat <- confusionMatrix(factor(convert_numeric_to_tiers(test_xgb_predictions)), test_data$Tier)
cfm <- as_tibble(conf_mat$table)
plot_confusion_matrix(cfm,
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n",
                      class_order = c("Reserve","Bench player","Above average","Star"))
# GET TEST ACCURACY FOR XGBOOST MODEL
accuracy_test_xgb <- mean(convert_numeric_to_tiers(test_xgb_predictions) == test_data$Tier)
weighted_accuracy_test_xgb <- sum(diag(conf_mat$table) * class_weights) / sum(t(conf_mat$table) * class_weights)

