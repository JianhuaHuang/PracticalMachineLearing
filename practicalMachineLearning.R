rm(list = ls())
sapply(c('caret', 'kernlab', 'car', 'ggplot2', 'ISLR', 'Hmisc'), require, 
       character.only = T)
#### Week 1
## spam email detect 'your' frequency
library(kernlab)
data(spam)
head(spam)
hist(spam[spam$type == 'nonspam', 'your'], prob = T, nclass = 80)
lines(density(spam[spam$type == 'nonspam','your']))
lines(density(spam[spam$type == 'spam', 'your']), col = 'blue')

prd <- ifelse(spam$your > .5, 'spam', 'nonspam')
table(prd, spam$type)/length(spam$type)

## in sample versus out of sample errors
set.seed(333)
smallSpam <-spam[sample(dim(spam)[1], size = 10), ]
spamLable <- (smallSpam$type == 'spam') * 1 + 1
plot(smallSpam$capitalAve, col = spamLable)

#### Week 2
## lectures
rm(list = ls())
gc()
set.seed(32343)
library(caret)
library(kernlab)
data(spam)

# return list with list = T, otherwise, return vector/matrix
inTrain <- createDataPartition(y = spam$type, p = .75, list = F) 
training <- spam[inTrain, ]
testing <- spam[-inTrain, ]
dim(training)

# spam example, fit a model with train function
# Here is the details of using your own model in train
# http://topepo.github.io/caret/custom_models.html
# http://topepo.github.io/caret/bytag.html
# use names(getModelInfo()) to get all possible models 
modelFit <- train(type ~ ., data = training, method = 'glm')
modelFit
modelFit$finalModel
predictions <- predict(modelFit, newdata = testing)
confusionMatrix(predictions, testing$type)

# spam example: K-fold
# returnTrain = F to return test datasets
set.seed(32323)
folds <- createFolds(y = spam$type, k = 10, list = TRUE, returnTrain = F)
sapply(folds, length)

# Resample: this may have duplicated selection
folds <- createResample(y = spam$type, times = 10, list = T)
sapply(folds, length)
head(folds[[1]], 10)
# 4  5  7  8  9 10 11 14 14 14 # some index selected multiple times

# Time slices, initialWindow and horizon are used to set the length of training
# and testing sets respectively.  
tme <- 1:1000
folds <- createTimeSlices(y = tme, initialWindow = 20, horizon = 10)

# Train options
args(train.default)
args(trainControl)

# plotting, example: wage data
library(ISLR)
library(ggplot2)
library(car)
library(Hmisc)
data(Wage)
inTrain <- createDataPartition(y = Wage$wage, p = .7, list = F)
training <- Wage[inTrain, ]
testing <- Wage[-inTrain, ]

featurePlot(x = training[, c('age', 'education', 'jobclass')], 
  y = training$wage, plot = 'pairs')
scatterplotMatrix(training[, c('age', 'education', 'jobclass', 'wage')])

gg <- ggplot(data = training, aes(x = age, y = wage, colour = education))
gg + geom_point() + geom_smooth(method = 'lm')
# or use the qplot, which is easier to write, but not as flexible as ggplot
qq <- qplot(age, wage, colour = education, data = training)
qq + geom_smooth(method = 'lm')

cutWage <- cut2(training$wage, g = 3) # cut wage into three groups
table(cutWage)
p1 <- ggplot(data = training, aes(x = cutWage, y = age, fill = cutWage))
p1 + geom_boxplot() + geom_jitter()

table(cutWage, training$jobclass)
# return the proportion of each elements in a given table
prop.table(table(cutWage, training$jobclass), 1)  

ggplot(data = training, aes(x = wage, colour = education)) + geom_density()

ggplot(data = training, aes(x = wage, colour = education, linetype = jobclass)) +
  geom_density() + 
  scale_linetype_manual(values = c(1, 3), labels = c('Industrial', 'Information'))

# preprocess
rm(list = ls())
gc()
sapply(c('caret', 'kernlab', 'car', 'ggplot2', 'ISLR', 'Hmisc'), require, 
  character.only = T)
data(spam)
inTrain <- createDataPartition(y = spam$type, p = .75, list = F)
training <- spam[inTrain, ]
testing <- spam[-inTrain, ]

hist(training$capitalAve)
mean(training$capitalAve)
sd(training$capitalAve)
trainCapAve <- training$capitalAve
trainCapAveS <- (trainCapAve - mean(trainCapAve)) / sd(trainCapAve)
sd(trainCapAveS)

# for test data, we should also use the mean and sd of the train data to 
# standardize the data. So the mean and sd of testCapAveS will not be 0 and 1
# but they should be close to it. 
testCapAve <- testing$capitalAve
testCapAveS <- (testCapAve - mean(trainCapAve)) / sd (trainCapAve)
sd(testCapAveS)

# the preProcess function in caret automate the estimating of 
# transformation (centering, scaling etc) from training data set, 
# which can be  further used to transform any data (training or testing sets)
# with predict function. 
preObj <- preProcess(training[, -58], method = c('center', 'scale'))
trainingS <- predict(preObj, training[, -58])
apply(trainingS, 2, function(x) {
  c(M = mean(x), SD = sd(x))
})  # all columns are standardize with mean 0 and sd 1
trainCapAves <- trainingS$capitalAve
testCapAveS <- predict(preObj, testing[, -58])$capitalAve
mean(testCapAveS)

# pass preProcess to train function directly to preProcess all predictior data!
set.seed(32343)
modelFit <- train(type ~., data = training, preProcess = c('center', 'scale'),
  method = 'glm')

# Box-Cox transformation
# it use the Maximum Likelihood to estimate the parameter to transform the data
# so that the transformed data looks as normal as possible. 
preObj <- preProcess(training[, -58], method = 'BoxCox')
trainCapAveS <- predict(preObj, training[, -58])$capitalAve
par(mfrow = c(3, 2))
hist(trainCapAve)
qqnorm(trainCapAve)
hist(log(trainCapAve))
qqnorm(log(trainCapAve))
hist(trainCapAveS)
qqnorm(trainCapAveS)

# Imputing data
set.seed(13343)
# make some values NA
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1], size = 1, prob = .05) == 1
training$capAve[selectNA] <- NA

# Pre-processing: 5 nearest neighbor imputation, scaled, centered 
# the 'knnImpute' will automatically standardize the data
preObj <- preProcess(training[, -58], method = 'knnImpute')  
capAve <- predict(preObj, training[, -58])$capAve

capAveTruth <- training$capitalAve # standardizing it to compare with capAve
capAveTruth <- (capAveTruth - mean(capAveTruth)) / sd(capAveTruth)

quantile(capAve - capAveTruth)
quantile((capAve - capAveTruth)[selectNA])
quantile((capAve - capAveTruth)[!selectNA])

# raw data to covariates (predictors)
data(Wage)
inTrain <- createDataPartition(y = Wage$wage, p = .7, list = F)
training <- Wage[inTrain, ]
testing <- Wage[-inTrain, ]
table(training$jobclass)

# create a full set of dummy variables
dummies <- dummyVars(wage ~ jobclass, data = training)
head(predict(dummies, newdata = training))

# Removing zero covariates that have few frequency and variance
nearZeroVar(training, saveMetrics = T) # sex and region

# spline basis
bsBasis <- bs(training$age, df = 3)  # generate polynomial spline
lm1 <- lm(wage ~ bsBasis, data = training) # lm1 is the same as lm2
lm2 <- lm(wage ~ bs(age, df = 3), data = training)

plot(wage ~ age, data = training, pch = 19, cex = .5)
points(training$age, lm1$fitted.values, col = 'red', pch = 19)
points(training$age, lm2$fitted.values, col = 'blue', pch = 19)
points(testing$age, predict(lm1, newdata = testing), col = 'green', pch = 19)

length(predict(lm2, newdata = testing))

# preprocessing with principle component Analysis (PCA)
rm(list = ls())
data(spam)
inTrain <- createDataPartition(y = spam$type, p = .75, list = F)
training <- spam[inTrain, ]
testing <- spam[-inTrain, ]

M <- abs(cor(training[, -58]))
diag(M) <- 0
which(M > .9, arr.ind = T)

colnames(spam)[c(34, 32)]
plot(spam[, 34] ~ spam[, 32])

# combine the highly correlated variables
X <- .71 * training$num415 + .71 * training$num857
Y <- .71 * training$num415 - .71 * training$num857
plot(X, Y)

# PCA
smallSpam <- spam[, c(34, 32)]
prComp <- prcomp(smallSpam)
plot(prComp$x[, 1], prComp$x[, 2])
prComp$rotation

typeColor <- (spam$type == 'spam') * 1 + 1
prComp <- prcomp(log10(spam[, -58] + 1))
plot(prComp$x[, 1], prComp$x[, 2], col = typeColor)

# PCA with caret
preProc <- preProcess(log10(spam[, -58] + 1), method = 'pca', pcaComp = 2)
spamPC <- predict(preProc, log10(spam[, -58] + 1))
plot(spamPC[, 1], spamPC[, 2], col = typeColor)

# build the modle with training set PCA
preProc <- preProcess(log10(training[, -58] + 1), method = 'pca', pcaComp = 2)
trainPC <- predict(preProc, newdata = log10(training[, -58] + 1))
modelFit <- train(training$type ~ ., data = trainPC, method = 'glm')

testPC <- predict(preProc, newdata = log10(testing[, -58] + 1))
confusionMatrix(testing$type, predict(modelFit, newdata = testPC))

# or put the PCA preProcess inside train function directly
# but how to set the pcaComp arguments with this method???
modelFit <- train(type ~ ., data = training, preProcess = 'pca',
                  method = 'glm')
confusionMatrix(testing$type, predict(modelFit, newdata = testing))

# Predicting with simple linear regression
data(faithful)
set.seed(333)
inTrain <- createDataPartition(y = faithful$waiting, p = .5, list = F)
trainFaith <- faithful[inTrain, ]
testFaith <- faithful[-inTrain, ]
plot(trainFaith$waiting, trainFaith$eruptions)
lm1 <- lm(eruptions ~ waiting, data = trainFaith)
lines(trainFaith$waiting, lm1$fitted)

# the lecture is wrong! it calculates the root sum square error
# use the mean instead of sum for during the calculation.
sqrt(mean(lm1$residual^2))  # root mean square error (RMSE)
sqrt(sum(lm1$residual^2)/lm1$df.residual)  # sigma (residual standard error) 
# equal to summary(lm1)$sigma

pred1 <- predict(lm1, newdata = testFaith, interval = 'prediction')
ord <- order(testFaith$waiting)
plot(eruptions ~ waiting, data = testFaith)
matlines(testFaith$waiting[ord], pred1[ord, ], type = 'l')  # plot columns of matrix

# same process with caret
lm2 <- train(eruptions ~ waiting, data = trainFaith, method = 'lm')
summary(lm2$finalModel)
summary(lm2)

# predicting with multiple covariate regression
data(Wage)
Wage <- subset(Wage, select = -c(logwage))
inTrain <- createDataPartition(y = Wage$wage, p = .7, list = F)
training <- Wage[inTrain, ]
testing <- Wage[-inTrain, ]
modFit <- train(wage ~ age + jobclass + education, method = 'lm', data = training)
# lm1 <- lm(wage ~ age + jobclass + education, data = training)
finMod <- modFit$finalModel
plot(finMod)

# color by variables not used in the model to find potential trends
ggplot(data = training, 
       aes(x = finMod$fitted, y = finMod$residuals, colour = race)) + 
  geom_point() + geom_smooth(method = 'lm')

# plot by index to find potential trends
plot(finMod$residuals)

# plot y again y hat
qplot(wage, finMod$fitted.values, data = training)

## Quiz 2
# Q1
library(AppliedPredictiveModeling)
library(caret)
data(AlzheimerDisease)

adData = data.frame(predictors)
trainIndex = createDataPartition(diagnosis,p=0.5,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]

adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]

# Q2
rm(list = ls())
gc()
library(AppliedPredictiveModeling)
library(caret)
library(Hmisc)
library(car)

data(concrete)
set.seed(975)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
plot(mixtures$CompressiveStrength, col = ((mixtures$FlyAsh > 0) + 1))
plot(mixtures$CompressiveStrength, col = cut2(mixtures$Age, g = 5))
plot(1:nrow(training), training$testing)
index <- seq_along(1:nrow(training))
ggplot(data = training, aes(x = index, y = CompressiveStrength)) + geom_point() + 
  theme_bw()

scatterplotMatrix(mixtures)
cut2(mixtures$FlyAsh, g = 5)

# Q3
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(975)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
hist(log(mixtures$Superplasticizer))

# Q4
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
trainIL <- training[, grep('^IL', colnames(training))]
preProc <- preProcess(trainIL, method = 'pca', thresh = .8)
preProc$rotation  # check the number of PC to get 80% variance
trainPC <- predict(preProc, newdata = trainIL)

# the result is different from using prcomp function
trainPC <- prcomp(trainIL)
summary(trainPC)

# Q5
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

trainIL <- training[, c(1, grep('^IL', colnames(training)))]
testIL <- testing[, c(1, grep('^IL', colnames(training)))]
mod1 <- train(diagnosis ~ ., data = trainIL, method = 'glm')
confusionMatrix(testIL$diagnosis, predict(mod1, newdata = testIL))

prePC <- preProcess(trainIL[, -1], method = 'pca', thresh = .8)
trainPC <- predict(prePC, newdata = trainIL[, -1])
testPC <- predict(prePC, newdata = testIL[, -1])
mod2 <- train(trainIL$diagnosis ~ ., data = trainPC, method = 'glm')
confusionMatrix(testIL$diagnosis, predict(mod2, newdata = testPC))

# or do it in one steop by incorporating the preProcess and thresh in train 
mod3 <- train(diagnosis ~ ., data = trainIL, preProcess = 'pca', method = 'glm',
              trControl = trainControl(preProcOptions = list(thresh = 0.8)))
confusionMatrix(testIL$diagnosis, predict(mod3, newdata = testIL))


#### Week 3
## predicting with tree
rm(list = ls())
library(rattle)
data(iris)
names(iris)
table(iris$Species)
ggplot(data = iris, aes(x = Petal.Width, y = Sepal.Width, colour = Species)) +
  geom_point()
inTrain <- createDataPartition(y = iris$Species, p = .7, list = F)
training <- iris[inTrain, ]
testing <- iris[-inTrain, ]

modFit <- train(Species ~ ., data = training, method = 'rpart')
print(modFit$finalModel)

plot(modFit$finalModel, uniform = T)
text(modFit$finalModel, use.n = T, all = T, cex = .8)

fancyRpartPlot(modFit$finalModel)

predict(modFit, newdata = testing)
confusionMatrix(testing$Species, predict(modFit, newdata = testing))

## bagging (Bootstrap aggregating)
library(ElemStatLearn)
data(ozone, package = 'ElemStatLearn')
ozone <- ozone[order(ozone$ozone), ]
head(ozone)

ll <- matrix(NA, nrow = 10, ncol = 155)
for(i in 1:10) {
  ss <- sample(1:nrow(ozone), replace = T)
  ozone0 <- ozone[ss, ]
  ozone0 <- ozone0[order(ozone0$ozone), ]
  loess0 <- loess(temperature ~ ozone, data = ozone0, span = .2)
  ll[i, ] <- predict(loess0, newdata = data.frame(ozone = 1:155))
}

plot(ozone$ozone, ozone$temperature, pch = 19, cex = .5)

for(i in 1:10) {
  lines(1:155, ll[i, ], col = 'grey')
}

lines(1:155, colMeans(ll), col = 'red', lwd = 2)

# bagging in caret package
predictors <- data.frame(ozone = ozone$ozone)
temperature = ozone$temperature
treebag <- bag(predictors, temperature, B = 10, 
  bagControl = bagControl(fit = ctreeBag$fit, predict = ctreeBag$pred,
    aggregate = ctreeBag$aggregate))

# points the fit with one of the model (red) and all model (blue) mean respectively
plot(temperature ~ ozone, data = ozone, col = 'lightgrey', pch = 19)
points(ozone$ozone, predict(treebag$fits[[1]]$fit, predictors), col = 'red')
points(ozone$ozone, predict(treebag, predictors), pch = 19, col = 'blue')

ctreeBag$fit
ctreeBag$pred

## random forest
# overfitting may be a problem, use cross-validation to avoid it
# check the rfcv function
data(iris)
inTrain <- createDataPartition(y = iris$Species, p = .7, list = F)
training <- iris[inTrain, ]
testing <- iris[-inTrain, ]

# prox argument used for classCenter function
modFit <- train(Species ~ ., data = training, method = 'rf', prox = T) 
modFit
getTree(modFit$finalModel, k = 2)

irisP <- classCenter(training[, c(3, 4)], training$Species, modFit$finalModel$prox)
irisP <- as.data.frame(irisP)
irisP$Species <- rownames(irisP)

ggplot(data = training, aes(x = Petal.Width, y = Petal.Length, colour = Species)) +
  geom_point() + 
  geom_point(data = irisP, aes(x = Petal.Width, y = Petal.Length, colour = Species),
    size = 9, shape = 4)

qplot(Petal.Width, Petal.Length, col = Species, data = training) + 
  geom_point(aes(x = Petal.Width, y = Petal.Length, col = Species),
    data = irisP, shape = 4)

pred <- predict(modFit, testing)
testing$predRight <- pred == testing$Species
table(pred, testing$Species)
confusionMatrix(pred, testing$Species)
qplot(Petal.Width, Petal.Length, colour = predRight, data = testing)

## boosting
rm(list = ls())
data(Wage)
Wage <- subset(Wage, select = - c(logwage))
inTrain <- createDataPartition(y = Wage$wage, p = .7, list = F)
training <- Wage[inTrain, ]
testing <- Wage[-inTrain, ]

# gbm is boosting with trees; mboost is model based boosting; 
modFit <- train(wage ~ ., data = training, method = 'gbm', verbose = F)
qplot(predict(modFit, testing), testing$wage)

## Model based prediction, a typical approach is to apply Bayes theorem
# Linear discriminant analysis 
# Quadratic discrimant analysis 
# Model based prediction 
# Naive Bayes
data(iris)
inTrain <- createDataPartition(y = iris$Species, p = .7, list = F)
training <- iris[inTrain, ]
testing <- iris[-inTrain, ]
modlda <- train(Species ~ ., data = training, method = 'lda')
modnb <- train(Species ~ ., data = training, method = 'nb')

plda <- predict(modlda, newdata = testing)
pnb <- predict(modnb, newdata = testing)

confusionMatrix(testing$Species, plda)
confusionMatrix(testing$Species, pnb)

equalPred <- plda == pnb
qplot(Petal.Width, Sepal.Width, colour = equalPred, data = testing)


## Quiz 3
# Q1
rm(list = ls())
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)

inTrain <- segmentationOriginal$Case == 'Train'
training <- segmentationOriginal[inTrain, ]
testing <- segmentationOriginal[-inTrain, ]
set.seed(125)
modCart <- train(Class ~ ., data = training, method = 'rpart')
fancyRpartPlot(modCart$finalModel)

# testing1 <- testing[1:4, ]
# testing1[!is.na(testing1)] <- NA
# testing1$TotalIntenCh2 <- c(23000, 50000, 57000, NA)
# testing1$FiberWidthCh1 <- c(10, 10, 8, 8)
# testing1$PerimStatusCh1 <- c(2, NA, NA, 2)
# testing1$VarIntenCh4 <- c(NA, 100, 100, 100)
# 
# predict(modCart, newdata = testing1, na.action = na.exclude)
# predict(modCart, newdata = testing)
# 
# confusionMatrix(testing$Class, predict(modCart, newdata = testing))


# Q3
rm(list = ls())
library(pgmm)
data(olive)
olive = olive[,-1]

modTree <- train(Area ~ ., data = olive, method = 'rpart')
fancyRpartPlot(modTree$finalModel)

predict(modTree, newdata = as.data.frame(t(colMeans(olive))))

# Q4
# refer to this link to calculate the accuracy/miscalculation rate:
# http://stackoverflow.com/questions/23806556/
# caret-train-predicts-very-different-then-predict-glm
rm(list = ls())
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

set.seed(13234)
modFit <- train(chd ~ age + alcohol + obesity + tobacco + typea + ldl,
  data = trainSA, method = 'glm', family = binomial)

modFit$finalModel$fitted.values
fitPredt <- ifelse(predict(modFit, newdata = trainSA) > .5, 1, 0)
confusionMatrix(trainSA$chd, fitPredt)

fitTest <- ifelse(predict(modFit, newdata = testSA) > .5, 1, 0)
confusionMatrix(testSA$chd, fitTest)

# Q5
rm(list = ls())
library(ElemStatLearn)
data(vowel.train)
data(vowel.test) 

vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)

set.seed(33883)
modFit <- train(y ~ ., data = vowel.train, method = 'rf', importance = T)

varImp(modFit, scale = F)

#### Week 4
## Regularized regression
# tradeoff between bias and variance: ridge, lasso, relaxo
data(prostate)
str(prostate)
head(prostate)

small <- prostate[1:5, ]
lm(lpsa ~ ., data = small)

## combining predictors (model ensembling)
# approaches for combining classifiers: 
# 1. bagging, boosting, random forests (usually combining similar classifiers)
# 2. combining different classifiers: model stacking, model ensembling
data(Wage)
Wage <- subset(Wage, select = -(logwage))

# create a building data set (training and testing) and validation set
inBuild <- createDataPartition(y = Wage$wage, p = .7, list = F)
buildData <- Wage[inBuild, ]
validation <- Wage[-inBuild, ]

inTrain <- createDataPartition(y = buildData$wage, p = .7, list = F)
training <- buildData[inTrain, ]
testing <- buildData[-inTrain, ]

# build two different models
mod1 <- train(wage ~ ., data = training, method = 'glm')
mod2 <- train(wage ~ ., data = training, method = 'rf', 
  trControl = trainControl(method = 'cv'), number = 3)  # cv rf

# fit a model that combines predictors with testing data
pred1 <- predict(mod1, newdata = testing)
pred2 <- predict(mod2, newdata = testing)
qplot(pred1, pred2, colour = wage, data = testing)

preDF <- data.frame(pred1, pred2, wage = testing$wage)
combModFit <- train(wage ~ ., data = preDF, method = 'gam')

combPred <- predict(combModFit, newdata = preDF)

sqrt(sum((pred1 - testing$wage)^2))
sqrt(sum((pred2 - testing$wage)^2))
sqrt(sum((combPred - testing$wage)^2))

# predict on validation data set
pred1V <- predict(mod1, newdata = validation)
pred2V <- predict(mod2, newdata = validation)
predVDF <- data.frame(pred1 = pred1V, pred2 = pred2V, wage = validation$wage)
combPredV <- predict(combModFit, newdata = predVDF)

sqrt(sum((pred1V - validation$wage)^2))
sqrt(sum((pred2V - validation$wage)^2))
sqrt(sum((combPredV - validation$wage)^2))

## forecasting
library(quantmod)
from.dat <- as.Date('01/03/08', format = '%m/%d/%y')
to.dat <- as.Date('12/31/13', format = '%m/%d/%y')
getSymbols('GOOG', src = 'google', from = from.dat, to = to.dat)
head(GOOG)
GOOG <- subset(GOOG, select = -GOOG.Volume)

mGoog <- to.monthly(GOOG)
googOpen <- Op(mGoog)
ts1 <- ts(googOpen, frequency = 12)
plot(ts1)
plot(decompose(ts1))  # decompose the time series trends

ts1Train <- window(ts1, start = 1, end = 5)
ts1Test <- window(ts1, start = 5, end = (7 - .01))
ts1Test

## unsupervised prediction
rm(list = ls())
data(iris)
inTrain <- createDataPartition(y = iris$Species, p = .7, list = F)
training <- iris[inTrain, ]
testing <- iris[-inTrain, ]
kMeans1 <- kmeans(subset(training, select = -Species), centers = 3)
training$clusters <- as.factor(kMeans1$cluster)
qplot(Petal.Width, Petal.Length, colour = clusters, data = training)

table(kMeans1$cluster, training$Species)

modFit <- train(clusters ~ ., data = training, method = 'rpart')
table(predict(modFit, newdata = training), training$Species)


testClusterPred <- predict(modFit, newdata = testing)
table(testClusterPred, testing$Species)


## Quiz 4
# Q1
rm(list = ls())
library(ElemStatLearn)
data(vowel.train)
data(vowel.test) 

vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)

set.seed(33883)
modRf <- train(y ~ ., data = vowel.train, method = 'rf')
modGbm <- train(y ~ ., data = vowel.train, method = 'gbm', verbose = F)

confusionMatrix(vowel.train$y, predict(modRf, newdata = vowel.train))
confusionMatrix(vowel.train$y, predict(modGbm, newdata = vowel.train))

confusionMatrix(vowel.test$y, predict(modRf, newdata = vowel.test))
confusionMatrix(vowel.test$y, predict(modGbm, newdata = vowel.test))

# Q2
rm(list = ls())
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

set.seed(62433)
modRf <- train(diagnosis ~ ., data = training, method = 'rf')
modGbm <- train(diagnosis ~ ., data = training, method = 'gbm', verbose = F)
modLda <- train(diagnosis ~ ., data = training, method = 'lda')

predTrainRf <- predict(modRf, newdata = training)
predTrainGbm <- predict(modGbm, newdata = training)
predTrainLda <- predict(modLda, newdata = training)

predTestRf <- predict(modRf, newdata = testing)
predTestGbm <- predict(modGbm, newdata = testing)
predTestLda <- predict(modLda, newdata = testing)

confusionMatrix(predTrainRf, training$diagnosis)
confusionMatrix(predTrainGbm, training$diagnosis)
confusionMatrix(predTrainLda, training$diagnosis)

confusionMatrix(predTestRf, testing$diagnosis)
confusionMatrix(predTestGbm, testing$diagnosis)
confusionMatrix(predTestLda, testing$diagnosis)

predDF <- data.frame(predTestRf, predTestGbm, predTestLda, 
                     diagnosis = testing$diagnosis)

modCom <- train(diagnosis ~ ., data = predDF, method = 'rf')

confusionMatrix(predDF$diagnosis, predict(modCom, newdata = predDF))

# Q3
rm(list = ls())
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(233)
modFit <- train(CompressiveStrength ~ ., data = training, method = 'lasso')
plot(modFit$finalModel)

# Q4
rm(list = ls())
library(lubridate)  # For year() function below
library(forecast)
library(quantmod)
dat = read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)

# Fit a model using the bats() function in the forecast package to the training 
# time series. Then forecast this model for the remaining time points. For how 
# many of the testing points is the true value within the 95% prediction 
# interval bounds?
# fit a model
fit <- bats(tstrain)

# check how long the test set is, so you can predict beyond trainign
h <- dim(testing)[1]

# forecast the model for remaining time points
fcast <- forecast(fit, level = 95, h = h)

sum(fcast$lower <= testing$visitsTumblr & fcast$upper >= testing$visitsTumblr)/h


# Q5
rm(list = ls())
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(325)
modFit <- train(CompressiveStrength ~ ., data = training, method = 'svmLinear')
pred <- predict(modFit, newdata = testing)
sqrt(mean((pred - testing$CompressiveStrength)^2))

# the results using svm function is very different from train function!!!
library(e1071)
modSvm <- svm(CompressiveStrength ~ ., data = training)
predSvm <- predict(modSvm, newdata = testing)
sqrt(mean((predSvm - testing$CompressiveStrength)^2))




