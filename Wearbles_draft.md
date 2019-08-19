---
title: "Wearables_draft"
author: "Czuee Morey"
date: "8/6/2019"
output: 
  html_document: 
    keep_md: yes
---

# Wearables activity prediciton




## Business Question

#### What are we trying to predict?

The classe variable which consists of 5 categories defining how well the exercises were carried out by 6 participants. We have a number of sensor measurements that can be used to predict this variable.

#### What type of problem is it?

This is a multivariate prediction problem in which we have to build a model. Describe how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

#### What type of data do we have? 
The data is in csv format. It presents a header row with the column names. The predictor variable is categorical.


## Read the files


```r
training <- read.csv("pml-training.csv", header = TRUE, na.strings = c("NA", "", "#DIV/0!"))
testing  <- read.csv("pml-testing.csv", header = TRUE, na.strings = c("NA", "", "#DIV/0!"))

dim(training); dim(testing)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```
I cleaned up the missing values, NAs and #DIV/0! while loading the files. There are a lot of missing values 19,000+, so imputation is not possible. 

##Exploratory data analysis

### Lets study the response variable
The response variable is classe. According to the text,
*Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz5vuWJGhXP*


```r
qplot(training$classe, ylab = "Frequency" , xlab = "Class", main = "Exercise accuracy")
```

![](Wearbles_draft_files/figure-html/unnamed-chunk-2-1.png)<!-- -->
Class A which is the exercise done correctly is the most frequent. 

### Understanding the variables 

The publication mentioned on the website has details about the features.

- Euler angles (roll, pitch and yaw)
- For Euler angles of each of 4 sensor, eight features calculated: mean, variance, standard deviation, max, min, amplitude, kurtosis and skewness (96 features)
-  raw accelerometer, gyroscope and magnetometer readings, for each sensor (12 features)
- 6 users



```r
descr <- dlookr::describe(training)
```

```
## Registered S3 method overwritten by 'xts':
##   method     from
##   as.zoo.xts zoo
```

```
## Registered S3 method overwritten by 'quantmod':
##   method            from
##   as.zoo.data.frame zoo
```

```
## Registered S3 methods overwritten by 'car':
##   method                          from
##   influence.merMod                lme4
##   cooks.distance.influence.merMod lme4
##   dfbeta.influence.merMod         lme4
##   dfbetas.influence.merMod        lme4
```

```r
descr[which(descr$skewness > 10), ]
```

```
## # A tibble: 16 x 26
##    variable     n    na     mean      sd se_mean     IQR skewness kurtosis
##    <chr>    <dbl> <dbl>    <dbl>   <dbl>   <dbl>   <dbl>    <dbl>    <dbl>
##  1 amplitu~   406 19216   3.77   2.53e+1 1.25e+0   1.78      13.9     195.
##  2 stddev_~   406 19216   1.34   1.03e+1 5.11e-1   0.6       15.1     238.
##  3 var_yaw~   406 19216 107.     1.66e+3 8.22e+1   0.465     17.3     315.
##  4 var_rol~   406 19216 417.     2.01e+3 9.96e+1 221.        10.9     134.
##  5 kurtosi~   401 19221   0.452  3.19e+0 1.59e-1   1.62      12.8     215.
##  6 kurtosi~   404 19218   0.286  3.30e+0 1.64e-1   1.30      12.5     200.
##  7 max_yaw~   401 19221   0.450  3.19e+0 1.59e-1   1.6       12.8     215.
##  8 min_yaw~   401 19221   0.450  3.19e+0 1.59e-1   1.6       12.8     215.
##  9 var_acc~   406 19216   4.39   1.35e+1 6.70e-1   3.06      12.2     195.
## 10 gyros_d~ 19622     0   0.0461 6.10e-1 4.35e-3   0.35      31.7    2682.
## 11 gyros_d~ 19622     0  -0.129  2.29e+0 1.63e-2   0.34     136.    18859.
## 12 kurtosi~   322 19300  -0.689  2.51e+0 1.40e-1   0.781     13.6     217.
## 13 max_yaw~   322 19300  -0.689  2.52e+0 1.40e-1   0.80      13.6     217.
## 14 min_yaw~   322 19300  -0.689  2.52e+0 1.40e-1   0.80      13.6     217.
## 15 gyros_f~ 19622     0   0.0752 3.10e+0 2.21e-2   3.08      51.3    5152.
## 16 gyros_f~ 19622     0   0.151  1.75e+0 1.25e-2   0.67     116.    15277.
## # ... with 17 more variables: p00 <dbl>, p01 <dbl>, p05 <dbl>, p10 <dbl>,
## #   p20 <dbl>, p25 <dbl>, p30 <dbl>, p40 <dbl>, p50 <dbl>, p60 <dbl>,
## #   p70 <dbl>, p75 <dbl>, p80 <dbl>, p90 <dbl>, p95 <dbl>, p99 <dbl>,
## #   p100 <dbl>
```

- Many of the variables have lots of NAs. For exampe, kurtosis_yaw_belt does not have any significant values. 
- The #DIV/0! are probably computation errors. Also, the variables where DIV/0 occur do not have other values and hence imputation is not possible.
- Readings for each of the individuals were taken only at a certain timepoint. The raw time point part2 is uniformly distributed, while part 1 only has specific timepoints.
- Some of the variables have high skewness & kurtosis
- Time of doing exercise and user names should not be important to make a prediction for the exercise and hence can be removed.

### Feature Selection & preprocessing

Remove columns with NAs, and the users, timestamps which are not important.


```r
trainproc <- training[colSums(!is.na(training)) > 0]
trainproc <- trainproc[ ,-c(1:7)]
```


```r
trainproc <- trainproc[sapply(trainproc, function(x) !any(is.na(x)))]
dim(trainproc)
```

```
## [1] 19622    53
```

PCA with selected features

```r
typeColor <- as.numeric(trainproc$classe)
preProc <- preProcess(trainproc[,-53],method="pca",pcaComp=10)
trainPC <- predict(preProc, trainproc[,-53])

xyplot(PC1 ~ PC2, data = trainPC, groups =  typeColor, auto.key = list(columns = 5)) #Colored by classe
```

![](Wearbles_draft_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

```r
xyplot(PC1 ~ PC2, data = trainPC, groups =  as.numeric(training$user_name),auto.key = list(columns = 6)) #Colored by user name
```

![](Wearbles_draft_files/figure-html/unnamed-chunk-6-2.png)<!-- -->
The data seems to be clearly clustered according to the different users. However, there also seems to be some clustering according to the classe variable. These patterns can be picked up using various training methods.

## Modeling

### Validation set
Creating a validation set to test out of sample error once I find suitable models.

```r
inTrain <- createDataPartition(trainproc$classe, p = 0.8, list = FALSE)

trainNum <- trainproc[inTrain,]
validation <- trainproc[-inTrain,]
dim(trainNum);dim(validation)
```

```
## [1] 15699    53
```

```
## [1] 3923   53
```

### Training on various predictors

```r
set.seed(123)
registerDoParallel(4) 
getDoParWorkers()
```

```
## [1] 4
```

```r
my_control <- trainControl(method = "cv", # for "cross-validation"
                           number = 3, # number of k-folds
                           savePredictions = "final",
                           allowParallel = TRUE)

fit1 <- train(classe ~ ., 
              data = trainNum,
              method = c("rf"),
              trControl = my_control,
              )
fit1$results$Accuracy
```

```
## [1] 0.9905088 0.9899992 0.9830567
```
Accuracy is 0.99


```r
trainpred <- predict(fit1, newdata = trainNum)

randomForest::varImpPlot(fit1$finalModel)
```

![](Wearbles_draft_files/figure-html/unnamed-chunk-9-1.png)<!-- -->


```r
fit2 <- train(classe ~ ., 
              data = trainNum,
              method = c("lda"),
              trControl = my_control,
              )

fit2
```

```
## Linear Discriminant Analysis 
## 
## 15699 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 10466, 10466, 10466 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.7041213  0.6255193
```


```r
set.seed(123)
registerDoParallel(4) 
getDoParWorkers()
```

```
## [1] 4
```

```r
fit3 <- train(classe ~ ., 
              data = trainNum,
              method = c("xgbTree"),
              trControl = my_control,
              )
fit3
```

```
## eXtreme Gradient Boosting 
## 
## 15699 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 10468, 10465, 10465 
## Resampling results across tuning parameters:
## 
##   eta  max_depth  colsample_bytree  subsample  nrounds  Accuracy 
##   0.3  1          0.6               0.50        50      0.8066125
##   0.3  1          0.6               0.50       100      0.8597381
##   0.3  1          0.6               0.50       150      0.8904403
##   0.3  1          0.6               0.75        50      0.8029826
##   0.3  1          0.6               0.75       100      0.8615210
##   0.3  1          0.6               0.75       150      0.8891658
##   0.3  1          0.6               1.00        50      0.8031100
##   0.3  1          0.6               1.00       100      0.8596738
##   0.3  1          0.6               1.00       150      0.8868098
##   0.3  1          0.8               0.50        50      0.8069935
##   0.3  1          0.8               0.50       100      0.8631130
##   0.3  1          0.8               0.50       150      0.8920966
##   0.3  1          0.8               0.75        50      0.8038095
##   0.3  1          0.8               0.75       100      0.8632405
##   0.3  1          0.8               0.75       150      0.8905675
##   0.3  1          0.8               1.00        50      0.8041924
##   0.3  1          0.8               1.00       100      0.8603746
##   0.3  1          0.8               1.00       150      0.8882745
##   0.3  2          0.6               0.50        50      0.9143912
##   0.3  2          0.6               0.50       100      0.9543925
##   0.3  2          0.6               0.50       150      0.9711446
##   0.3  2          0.6               0.75        50      0.9123529
##   0.3  2          0.6               0.75       100      0.9554753
##   0.3  2          0.6               0.75       150      0.9721002
##   0.3  2          0.6               1.00        50      0.9089769
##   0.3  2          0.6               1.00       100      0.9533103
##   0.3  2          0.6               1.00       150      0.9707627
##   0.3  2          0.8               0.50        50      0.9154104
##   0.3  2          0.8               0.50       100      0.9569410
##   0.3  2          0.8               0.50       150      0.9735015
##   0.3  2          0.8               0.75        50      0.9182767
##   0.3  2          0.8               0.75       100      0.9564949
##   0.3  2          0.8               0.75       150      0.9736289
##   0.3  2          0.8               1.00        50      0.9159837
##   0.3  2          0.8               1.00       100      0.9552215
##   0.3  2          0.8               1.00       150      0.9712725
##   0.3  3          0.6               0.50        50      0.9631195
##   0.3  3          0.6               0.50       100      0.9847124
##   0.3  3          0.6               0.50       150      0.9900631
##   0.3  3          0.6               0.75        50      0.9626729
##   0.3  3          0.6               0.75       100      0.9847763
##   0.3  3          0.6               0.75       150      0.9907636
##   0.3  3          0.6               1.00        50      0.9626737
##   0.3  3          0.6               1.00       100      0.9845217
##   0.3  3          0.6               1.00       150      0.9907000
##   0.3  3          0.8               0.50        50      0.9645205
##   0.3  3          0.8               0.50       100      0.9845851
##   0.3  3          0.8               0.50       150      0.9906361
##   0.3  3          0.8               0.75        50      0.9643296
##   0.3  3          0.8               0.75       100      0.9850948
##   0.3  3          0.8               0.75       150      0.9905090
##   0.3  3          0.8               1.00        50      0.9636286
##   0.3  3          0.8               1.00       100      0.9844577
##   0.3  3          0.8               1.00       150      0.9899356
##   0.4  1          0.6               0.50        50      0.8266767
##   0.4  1          0.6               0.50       100      0.8852800
##   0.4  1          0.6               0.50       150      0.9085305
##   0.4  1          0.6               0.75        50      0.8287155
##   0.4  1          0.6               0.75       100      0.8799930
##   0.4  1          0.6               0.75       150      0.9082122
##   0.4  1          0.6               1.00        50      0.8265502
##   0.4  1          0.6               1.00       100      0.8802483
##   0.4  1          0.6               1.00       150      0.9062371
##   0.4  1          0.8               0.50        50      0.8327938
##   0.4  1          0.8               0.50       100      0.8844525
##   0.4  1          0.8               0.50       150      0.9101874
##   0.4  1          0.8               0.75        50      0.8281429
##   0.4  1          0.8               0.75       100      0.8815862
##   0.4  1          0.8               0.75       150      0.9086579
##   0.4  1          0.8               1.00        50      0.8246400
##   0.4  1          0.8               1.00       100      0.8797394
##   0.4  1          0.8               1.00       150      0.9047093
##   0.4  2          0.6               0.50        50      0.9330545
##   0.4  2          0.6               0.50       100      0.9680872
##   0.4  2          0.6               0.50       150      0.9816552
##   0.4  2          0.6               0.75        50      0.9331817
##   0.4  2          0.6               0.75       100      0.9691060
##   0.4  2          0.6               0.75       150      0.9810818
##   0.4  2          0.6               1.00        50      0.9300605
##   0.4  2          0.6               1.00       100      0.9658579
##   0.4  2          0.6               1.00       150      0.9787251
##   0.4  2          0.8               0.50        50      0.9377679
##   0.4  2          0.8               0.50       100      0.9697432
##   0.4  2          0.8               0.50       150      0.9816548
##   0.4  2          0.8               0.75        50      0.9336278
##   0.4  2          0.8               0.75       100      0.9684695
##   0.4  2          0.8               0.75       150      0.9814002
##   0.4  2          0.8               1.00        50      0.9306338
##   0.4  2          0.8               1.00       100      0.9661129
##   0.4  2          0.8               1.00       150      0.9795529
##   0.4  3          0.6               0.50        50      0.9721006
##   0.4  3          0.6               0.50       100      0.9882797
##   0.4  3          0.6               0.50       150      0.9914643
##   0.4  3          0.6               0.75        50      0.9734379
##   0.4  3          0.6               0.75       100      0.9895534
##   0.4  3          0.6               0.75       150      0.9918465
##   0.4  3          0.6               1.00        50      0.9722916
##   0.4  3          0.6               1.00       100      0.9889165
##   0.4  3          0.6               1.00       150      0.9917829
##   0.4  3          0.8               0.50        50      0.9743297
##   0.4  3          0.8               0.50       100      0.9889801
##   0.4  3          0.8               0.50       150      0.9919102
##   0.4  3          0.8               0.75        50      0.9738202
##   0.4  3          0.8               0.75       100      0.9895534
##   0.4  3          0.8               0.75       150      0.9923559
##   0.4  3          0.8               1.00        50      0.9738840
##   0.4  3          0.8               1.00       100      0.9889800
##   0.4  3          0.8               1.00       150      0.9921012
##   Kappa    
##   0.7550109
##   0.8223963
##   0.8613043
##   0.7503938
##   0.8246834
##   0.8596884
##   0.7504921
##   0.8223721
##   0.8567091
##   0.7553950
##   0.8266768
##   0.8633792
##   0.7513232
##   0.8268820
##   0.8614538
##   0.7518692
##   0.8232602
##   0.8585327
##   0.8916865
##   0.9423073
##   0.9634973
##   0.8890875
##   0.9436696
##   0.9647067
##   0.8848578
##   0.9409448
##   0.9630184
##   0.8929729
##   0.9455278
##   0.9664800
##   0.8965636
##   0.9449611
##   0.9666414
##   0.8937118
##   0.9433570
##   0.9636624
##   0.9533385
##   0.9806618
##   0.9874310
##   0.9527868
##   0.9807435
##   0.9883166
##   0.9527831
##   0.9804211
##   0.9882362
##   0.9551203
##   0.9805013
##   0.9881550
##   0.9548790
##   0.9811460
##   0.9879947
##   0.9539948
##   0.9803399
##   0.9872689
##   0.7805891
##   0.8547696
##   0.8842102
##   0.7830767
##   0.8480670
##   0.8837951
##   0.7802696
##   0.8483728
##   0.8812946
##   0.7882505
##   0.8537027
##   0.8863001
##   0.7823540
##   0.8501061
##   0.8843718
##   0.7778897
##   0.8477971
##   0.8793712
##   0.9153159
##   0.9596338
##   0.9767955
##   0.9154844
##   0.9609155
##   0.9760675
##   0.9115236
##   0.9568129
##   0.9730891
##   0.9212728
##   0.9617256
##   0.9767934
##   0.9160499
##   0.9601149
##   0.9764729
##   0.9122638
##   0.9571351
##   0.9741348
##   0.9647113
##   0.9851748
##   0.9892027
##   0.9663991
##   0.9867859
##   0.9896866
##   0.9649503
##   0.9859804
##   0.9896062
##   0.9675281
##   0.9860602
##   0.9897668
##   0.9668826
##   0.9867863
##   0.9903309
##   0.9669684
##   0.9860613
##   0.9900089
## 
## Tuning parameter 'gamma' was held constant at a value of 0
## 
## Tuning parameter 'min_child_weight' was held constant at a value of 1
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were nrounds = 150, max_depth = 3,
##  eta = 0.4, gamma = 0, colsample_bytree = 0.8, min_child_weight = 1
##  and subsample = 0.75.
```

Both random forest and xgbtree give an accuracy of 0.99 on the training set. Let's see how they perform on the validation set.

### Out-of-sample accuracy

```r
valpred1 <- predict(fit1, newdata = validation)
valpred2 <- predict(fit2, newdata = validation)
valpred3 <- predict(fit3, newdata = validation)

confusionMatrix(valpred1, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    1    0    0    0
##          B    0  757    1    0    0
##          C    0    1  683    2    0
##          D    0    0    0  641    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9987         
##                  95% CI : (0.997, 0.9996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9984         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9974   0.9985   0.9969   1.0000
## Specificity            0.9996   0.9997   0.9991   1.0000   1.0000
## Pos Pred Value         0.9991   0.9987   0.9956   1.0000   1.0000
## Neg Pred Value         1.0000   0.9994   0.9997   0.9994   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1930   0.1741   0.1634   0.1838
## Detection Prevalence   0.2847   0.1932   0.1749   0.1634   0.1838
## Balanced Accuracy      0.9998   0.9985   0.9988   0.9984   1.0000
```

```r
#confusionMatrix(valpred2, validation$classe)
confusionMatrix(valpred3, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  759    0    0    0
##          C    0    0  683    0    0
##          D    0    0    1  642    0
##          E    0    0    0    1  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9995          
##                  95% CI : (0.9982, 0.9999)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9994          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9985   0.9984   1.0000
## Specificity            1.0000   1.0000   1.0000   0.9997   0.9997
## Pos Pred Value         1.0000   1.0000   1.0000   0.9984   0.9986
## Neg Pred Value         1.0000   1.0000   0.9997   0.9997   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1935   0.1741   0.1637   0.1838
## Detection Prevalence   0.2845   0.1935   0.1741   0.1639   0.1840
## Balanced Accuracy      1.0000   1.0000   0.9993   0.9991   0.9998
```

xgbtree has the best prediction accuracy on the validation dataset of 0.9967, as well as better per-class  statistics. Random forest is a close second.

## Test set prediction

```r
test.xgb <- predict(fit3, newdata = testing)
test.rf <- predict(fit1, newdata = testing)

test.xgb; test.rf
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The prediction using both RF & XGB tree is identical on the test set.
