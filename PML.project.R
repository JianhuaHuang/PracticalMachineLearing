rm(list = ls())
gc()
setwd('/NAS/jhuang/Projects/PracticalMachineLearning')
sapply(c('caret', 'rattle', 'ggplot2', 'doMC'), require, character.only = T)

buildData <- read.csv('pml-training.csv', row.names = 1, na.string = c(NA, ''))
validation <- read.csv('pml-testing.csv', row.names = 1, na.string = c(NA, ''))

naCol <- which(colSums(is.na(buildData)) > .9 * nrow(buildData))
buildData <- buildData[, -naCol]

nzv <- nearZeroVar(buildData, saveMetrics = T) 

validation <- validation[, -naCol]  # keep the same columns in validataion


inTrain <- createDataPartition(y = buildData$classe, p = .7, list = F)
training <- buildData[inTrain, ]
testing <- buildData[-inTrain, ]

dim(training)
dim(testing)
dim(validation)

set.seed(123)
modLda <- train(classe ~ ., data = training, method = 'lda')

modRpart <- train(classe ~ ., data = training, method = 'rpart')

cl <- makeCluster(12)
registerDoMC(cl)
modGbm <- train(classe ~ ., data = training, method = 'gbm', verbose = F,
  trControl = trainControl(## 10-fold CV, repeat 3 times
    method = "repeatedcv", number = 4, repeats = 3))
stopCluster(cl)

cl <- makeCluster(12)
registerDoMC(cl)
modRf <- train(classe ~ ., data = training, method = 'rf', 
  trControl = trainControl(method = 'cv', number = 10))
stopCluster(cl)

confusionMatrix(predict(modLda, newdata = testing), testing$classe)
confusionMatrix(predict(modRpart, newdata = testing), testing$classe)
confusionMatrix(predict(modGbm, newdata = testing), testing$classe)
confusionMatrix(predict(modRf, newdata = testing), testing$classe)

predict(modLda, validation)
predict(modRpart, validation)
predict(modGbm, validation)
predict(modRf, validation)

fancyRpartPlot(modRpart$finalModel)

varImp(modGbm)
qplot(roll_belt, pitch_forearm, colour = classe, data = testing)


pred <- predict(mod1, newdata = testing)
confusionMatrix(pred, testing$classe)

answers <- predict(mod1, newdata = validation)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("PML.project/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
