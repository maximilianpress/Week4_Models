Predicting blood donations 
===============================
## Maximilian Press

Various looks at what you can do with models, hopefully with an emphasis on parametric (GLM) models. 

Takes a statistical learning point of view on the problem.

This is a good resource: Dolph Schluter's R modeling pages. https://www.zoology.ubc.ca/~schluter/R/fit-model/


```r
require(visreg)
trans = read.csv('transfusion.data',header=T)
cor(trans)
```

```
##                                            Recency..months.
## Recency..months.                                  1.0000000
## Frequency..times.                                -0.1827455
## Monetary..c.c..blood.                            -0.1827455
## Time..months.                                     0.1606181
## whether.he.she.donated.blood.in.March.2007       -0.2798689
##                                            Frequency..times.
## Recency..months.                                  -0.1827455
## Frequency..times.                                  1.0000000
## Monetary..c.c..blood.                              1.0000000
## Time..months.                                      0.6349403
## whether.he.she.donated.blood.in.March.2007         0.2186334
##                                            Monetary..c.c..blood.
## Recency..months.                                      -0.1827455
## Frequency..times.                                      1.0000000
## Monetary..c.c..blood.                                  1.0000000
## Time..months.                                          0.6349403
## whether.he.she.donated.blood.in.March.2007             0.2186334
##                                            Time..months.
## Recency..months.                              0.16061809
## Frequency..times.                             0.63494027
## Monetary..c.c..blood.                         0.63494027
## Time..months.                                 1.00000000
## whether.he.she.donated.blood.in.March.2007   -0.03585441
##                                            whether.he.she.donated.blood.in.March.2007
## Recency..months.                                                          -0.27986887
## Frequency..times.                                                          0.21863344
## Monetary..c.c..blood.                                                      0.21863344
## Time..months.                                                             -0.03585441
## whether.he.she.donated.blood.in.March.2007                                 1.00000000
```
Look at the data a little (Figure 1).

```r
plot(trans,cex=.5)
```

![plot of chunk unnamed-chunk-2](figure/unnamed-chunk-2-1.png) 
Obviously some of these things are more meaningful than other things.  I will sorta naively fit the model based on everything, ignoring the possibility of interactions.

Using prediction to evaluate the model. I chose to randomly sample 500 observations to train, and test on the remaining 248.  


```r
# training set
trainindex = sample(1:748,500)
train = trans[trainindex,]
# test set
test = trans[!(1:nrow(trans) %in% trainindex),]

# some utility functions
source('roc.R')
```

First, fit a linear model, which is ok but not very interesting.

```r
plot(train$Frequency..times.,
	jitter(train$whether.he.she.donated.blood.in.March.2007),
	xlab='# donation events',ylab='donated in test period (jittered)', 
	cex = .5 )

# fit the model
linmod = lm(whether.he.she.donated.blood.in.March.2007 ~ 
	Frequency..times.,data = train)
str(linmod)
```

```
## List of 12
##  $ coefficients : Named num [1:2] 0.1507 0.0163
##   ..- attr(*, "names")= chr [1:2] "(Intercept)" "Frequency..times."
##  $ residuals    : Named num [1:500] -0.2 -0.248 -0.346 -0.265 -0.167 ...
##   ..- attr(*, "names")= chr [1:500] "12" "201" "445" "331" ...
##  $ effects      : Named num [1:500] -5.411 -2.055 -0.325 -0.253 -0.167 ...
##   ..- attr(*, "names")= chr [1:500] "(Intercept)" "Frequency..times." "" "" ...
##  $ rank         : int 2
##  $ fitted.values: Named num [1:500] 0.2 0.248 0.346 0.265 0.167 ...
##   ..- attr(*, "names")= chr [1:500] "12" "201" "445" "331" ...
##  $ assign       : int [1:2] 0 1
##  $ qr           :List of 5
##   ..$ qr   : num [1:500, 1:2] -22.3607 0.0447 0.0447 0.0447 0.0447 ...
##   .. ..- attr(*, "dimnames")=List of 2
##   .. .. ..$ : chr [1:500] "12" "201" "445" "331" ...
##   .. .. ..$ : chr [1:2] "(Intercept)" "Frequency..times."
##   .. ..- attr(*, "assign")= int [1:2] 0 1
##   ..$ qraux: num [1:2] 1.04 1
##   ..$ pivot: int [1:2] 1 2
##   ..$ tol  : num 1e-07
##   ..$ rank : int 2
##   ..- attr(*, "class")= chr "qr"
##  $ df.residual  : int 498
##  $ xlevels      : Named list()
##  $ call         : language lm(formula = whether.he.she.donated.blood.in.March.2007 ~ Frequency..times.,      data = train)
##  $ terms        :Classes 'terms', 'formula' length 3 whether.he.she.donated.blood.in.March.2007 ~ Frequency..times.
##   .. ..- attr(*, "variables")= language list(whether.he.she.donated.blood.in.March.2007, Frequency..times.)
##   .. ..- attr(*, "factors")= int [1:2, 1] 0 1
##   .. .. ..- attr(*, "dimnames")=List of 2
##   .. .. .. ..$ : chr [1:2] "whether.he.she.donated.blood.in.March.2007" "Frequency..times."
##   .. .. .. ..$ : chr "Frequency..times."
##   .. ..- attr(*, "term.labels")= chr "Frequency..times."
##   .. ..- attr(*, "order")= int 1
##   .. ..- attr(*, "intercept")= int 1
##   .. ..- attr(*, "response")= int 1
##   .. ..- attr(*, ".Environment")=<environment: R_GlobalEnv> 
##   .. ..- attr(*, "predvars")= language list(whether.he.she.donated.blood.in.March.2007, Frequency..times.)
##   .. ..- attr(*, "dataClasses")= Named chr [1:2] "numeric" "numeric"
##   .. .. ..- attr(*, "names")= chr [1:2] "whether.he.she.donated.blood.in.March.2007" "Frequency..times."
##  $ model        :'data.frame':	500 obs. of  2 variables:
##   ..$ whether.he.she.donated.blood.in.March.2007: int [1:500] 0 0 0 0 0 1 1 0 0 0 ...
##   ..$ Frequency..times.                         : int [1:500] 3 6 12 7 1 5 8 2 2 5 ...
##   ..- attr(*, "terms")=Classes 'terms', 'formula' length 3 whether.he.she.donated.blood.in.March.2007 ~ Frequency..times.
##   .. .. ..- attr(*, "variables")= language list(whether.he.she.donated.blood.in.March.2007, Frequency..times.)
##   .. .. ..- attr(*, "factors")= int [1:2, 1] 0 1
##   .. .. .. ..- attr(*, "dimnames")=List of 2
##   .. .. .. .. ..$ : chr [1:2] "whether.he.she.donated.blood.in.March.2007" "Frequency..times."
##   .. .. .. .. ..$ : chr "Frequency..times."
##   .. .. ..- attr(*, "term.labels")= chr "Frequency..times."
##   .. .. ..- attr(*, "order")= int 1
##   .. .. ..- attr(*, "intercept")= int 1
##   .. .. ..- attr(*, "response")= int 1
##   .. .. ..- attr(*, ".Environment")=<environment: R_GlobalEnv> 
##   .. .. ..- attr(*, "predvars")= language list(whether.he.she.donated.blood.in.March.2007, Frequency..times.)
##   .. .. ..- attr(*, "dataClasses")= Named chr [1:2] "numeric" "numeric"
##   .. .. .. ..- attr(*, "names")= chr [1:2] "whether.he.she.donated.blood.in.March.2007" "Frequency..times."
##  - attr(*, "class")= chr "lm"
```

```r
# things you can do with the fitted model object
abline(linmod)	# add the predicted function to the plot just generated
```

![Predictions of linear model (training only)](figure/unnamed-chunk-4-1.png) 

```r
# return various useful information about the model:
summary(linmod)	# print a lot of results, in semi-human-readable table
```

```
## 
## Call:
## lm(formula = whether.he.she.donated.blood.in.March.2007 ~ Frequency..times., 
##     data = train)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -0.7692 -0.2321 -0.1833 -0.1670  0.8330 
## 
## Coefficients:
##                   Estimate Std. Error t value Pr(>|t|)    
## (Intercept)        0.15073    0.02642   5.705 1.99e-08 ***
## Frequency..times.  0.01627    0.00332   4.902 1.28e-06 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.4192 on 498 degrees of freedom
## Multiple R-squared:  0.04603,	Adjusted R-squared:  0.04412 
## F-statistic: 24.03 on 1 and 498 DF,  p-value: 1.285e-06
```

```r
coef(linmod) 	# coefficients (parameters)
```

```
##       (Intercept) Frequency..times. 
##        0.15073350        0.01627434
```

```r
confint(linmod)	# confidence intervals
```

```
##                         2.5 %     97.5 %
## (Intercept)       0.098825340 0.20264165
## Frequency..times. 0.009751692 0.02279699
```

```r
resid(linmod)[1:10]	# residuals on the model -  printing out only first ten
```

```
##         12        201        445        331        409        306 
## -0.1995565 -0.2483795 -0.3460256 -0.2646539 -0.1670078  0.7678948 
##         62        662        148        126 
##  0.7190718 -0.1832822 -0.1832822 -0.2321052
```

```r
anova(linmod)	# anova table
```

```
## Analysis of Variance Table
## 
## Response: whether.he.she.donated.blood.in.March.2007
##                    Df Sum Sq Mean Sq F value    Pr(>F)    
## Frequency..times.   1  4.222  4.2221  24.031 1.285e-06 ***
## Residuals         498 87.496  0.1757                      
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

```r
# this would plot lots of model fit info, which may or may not be useful:
#plot(linmod)	# commented because verbose

# alternate visualization method
library(visreg)
visreg(linmod)
```

![Predictions of linear model (training only)](figure/unnamed-chunk-4-2.png) 

So we had a low p-value, which is good right? Problem solved, everyone go home.

Except this is obviously a really crappy model. This can be shown if we try to predict test values (new data that wasn't used to build the model, just plugging new values into the model function) and compare them to the actual values of the test outcome.


```r
linpred = predict(linmod,newdata=test)
linpredplot = plot(
	jitter(test$whether.he.she.donated.blood.in.March.2007), 
	linpred, 
	xlab='True value (jittered)', ylab='Predicted value', 
	xlim = c(-.2,1.2), ylim = c(0,1), cex = .5)

points( c(0,1), c(0,1), cex = 2, pch = 19 )
```

![Predictions vs. true values from linear model on test data](figure/unnamed-chunk-5-1.png) 

```r
prediction = cbind(linpred,test[,5])
a = ROC(prediction)
```

![ROC analysis of linear model on test](figure/unnamed-chunk-6-1.png) 

Not great. There are also some numerical summaries of model fit that various people use (besides $R^2$).


```r
# Akaike information criterion: -2ln( L(model) ) - 2*(num parameters)
AIC(linmod)	
```

```
## [1] 553.4305
```

```r
# stolen from https://www.kaggle.com/c/bioresponse/forums/t/1576/r-code-for-logloss
LogLoss = function(actual, predicted)	
# for explanation see https://en.wikipedia.org/wiki/Loss_function
	{
	result = -1/length(actual) * 
	(sum((actual*log(predicted)+(1-actual) *
	log(1-predicted))))
	return(result)
	}

# note that this makes use of training set
LogLoss( test$whether.he.she.donated.blood.in.March.2007, linpred )	
```

```
## [1] 0.5156009
```

```r
# AUC from the ROC curve above also is such a measure.
# you can even use a U-test to sort of evaluate the quality of the predictions:
wilcox.test( 
	linpred[test$whether.he.she.donated.blood.in.March.2007 == 1],
	linpred[test$whether.he.she.donated.blood.in.March.2007 == 0] 
	)
```

```
## 
## 	Wilcoxon rank sum test with continuity correction
## 
## data:  linpred[test$whether.he.she.donated.blood.in.March.2007 == 1] and linpred[test$whether.he.she.donated.blood.in.March.2007 == 0]
## W = 7092.5, p-value = 0.0004621
## alternative hypothesis: true location shift is not equal to 0
```


Try instead a logistic regression: a generalized linear model (GLM) of the family "binomial". That is, it expects the outcome variable (blood donation) to be distributed as a binomial (0/1) random variable. The predictor "generalizes" a linear fit using the logistic function to be able to make discrete 0/1 predictions.


```r
trainfit = glm(whether.he.she.donated.blood.in.March.2007 ~ Recency..months. 
	+ Frequency..times. + Monetary..c.c..blood. 
	+ Time..months.,family='binomial',data=train
	)

# do some predictions
predictor = predict.glm(trainfit,newdata=test)
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```r
prediction = cbind(predictor,test[,5])

# really crude look at prediction success
cor(prediction,method='spearman')
```

```
##           predictor          
## predictor 1.0000000 0.3753177
##           0.3753177 1.0000000
```

```r
a=ROC(prediction)
```

![Performance of naive logistic regression](figure/unnamed-chunk-8-1.png) 

So it's not great, but okay (i.e. AUC>0.5).  Specifically, the precision goes to hell very quickly.  Does stepwise regression help it by getting rid of spurious variables that are overfitting?


```r
stepfit = step(trainfit)
```

```
## Start:  AIC=484.09
## whether.he.she.donated.blood.in.March.2007 ~ Recency..months. + 
##     Frequency..times. + Monetary..c.c..blood. + Time..months.
## 
## 
## Step:  AIC=484.09
## whether.he.she.donated.blood.in.March.2007 ~ Recency..months. + 
##     Frequency..times. + Time..months.
## 
##                     Df Deviance    AIC
## <none>                   476.09 484.09
## - Time..months.      1   486.15 492.15
## - Frequency..times.  1   498.66 504.66
## - Recency..months.   1   506.65 512.65
```
So it looks like "Monetary" is pretty much colinear with "frequency", so no additional information (removing it seems to leave exactly the same model).  Also, recency does not seem to add much, so I am going to remove that.


```r
curated_fit = glm(whether.he.she.donated.blood.in.March.2007 ~ 
	Frequency..times. 
	+ Time..months.,family='binomial',
	data=train)
curated_prediction = predict.glm(curated_fit,newdata=test)

prediction = cbind(curated_prediction,test[,5])
```

```r
a=ROC(prediction)
```

![Performance of logistic regression with reduced model](figure/unnamed-chunk-11-1.png) 

Didn't really change much (Figure 3).  Lost a little AUC, but not much for removing 2 explanatory variables in slavish devotion to occam's razor.  Precision seems to fall apart a bit, though.  While logistic regression is nice and simple, it is not doing a super job, so I will move on to see if anything else does better.

## Naive Bayes
Naive Bayes is an attractively simple classification technique. It is similar to the initial logistic regression implemented above, because of its assumption of independence of predictor variables.  It uses a straightforward interpretation of Bayes' rule to compute probabilities of each variable belonging to each class.  While we only have a binary outcome, it is possible that NB will perform better for some reason.  


```r
require(e1071)
# this function wants response to be a factor
classifier = naiveBayes(train[,1:4],as.factor(train[,5]))
print(classifier)
```

```
## 
## Naive Bayes Classifier for Discrete Predictors
## 
## Call:
## naiveBayes.default(x = train[, 1:4], y = as.factor(train[, 5]))
## 
## A-priori probabilities:
## as.factor(train[, 5])
##     0     1 
## 0.758 0.242 
## 
## Conditional probabilities:
##                      Recency..months.
## as.factor(train[, 5])      [,1]     [,2]
##                     0 11.153034 8.266578
##                     1  5.669421 5.436587
## 
##                      Frequency..times.
## as.factor(train[, 5])     [,1]     [,2]
##                     0 4.923483 4.680473
##                     1 7.752066 7.597457
## 
##                      Monetary..c.c..blood.
## as.factor(train[, 5])     [,1]     [,2]
##                     0 1230.871 1170.118
##                     1 1938.017 1899.364
## 
##                      Time..months.
## as.factor(train[, 5])     [,1]     [,2]
##                     0 35.26385 24.35255
##                     1 33.61157 23.63908
```

```r
bayespredict = cbind(predict(classifier,test[,-5]),test[,5])
```

```r
a=ROC(bayespredict)
```

![Performance of Naive Bayes predictor](figure/unnamed-chunk-13-1.png) 
Well, it turns out that Bayesian statistics is not the answer to everything (Figure 4).  About the same as the reduced logistic regression model.  The curve is weirdly step-like, wonder what's going on there.  Perhaps because NB is specifying categorical cutoffs in the continuous data?

## Interaction effects
So far I have made the simplifying assumption that the variables are independent.  This obviously isn't the case.  Maybe what I am missing is interactions between variables, which contain something extra.  I will go back to the logistic regression model, except this time add interactions.

```r
interfit = glm(whether.he.she.donated.blood.in.March.2007 ~ Recency..months. * Frequency..times. * Time..months.,family='binomial',data=train)

interstep = step(interfit)
```

```
## Start:  AIC=483.27
## whether.he.she.donated.blood.in.March.2007 ~ Recency..months. * 
##     Frequency..times. * Time..months.
## 
##                                                    Df Deviance    AIC
## - Recency..months.:Frequency..times.:Time..months.  1   467.96 481.96
## <none>                                                  467.27 483.27
## 
## Step:  AIC=481.96
## whether.he.she.donated.blood.in.March.2007 ~ Recency..months. + 
##     Frequency..times. + Time..months. + Recency..months.:Frequency..times. + 
##     Recency..months.:Time..months. + Frequency..times.:Time..months.
## 
##                                      Df Deviance    AIC
## <none>                                    467.96 481.96
## - Recency..months.:Time..months.      1   471.12 483.12
## - Frequency..times.:Time..months.     1   471.13 483.13
## - Recency..months.:Frequency..times.  1   471.35 483.35
```

```r
summary(interstep)
```

```
## 
## Call:
## glm(formula = whether.he.she.donated.blood.in.March.2007 ~ Recency..months. + 
##     Frequency..times. + Time..months. + Recency..months.:Frequency..times. + 
##     Recency..months.:Time..months. + Frequency..times.:Time..months., 
##     family = "binomial", data = train)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.8177  -0.7329  -0.4779  -0.2746   2.5129  
## 
## Coefficients:
##                                      Estimate Std. Error z value Pr(>|z|)
## (Intercept)                        -0.6610710  0.3438585  -1.923 0.054542
## Recency..months.                   -0.1238853  0.0372192  -3.329 0.000873
## Frequency..times.                   0.2761544  0.0638677   4.324 1.53e-05
## Time..months.                      -0.0273451  0.0097641  -2.801 0.005101
## Recency..months.:Frequency..times. -0.0080154  0.0043784  -1.831 0.067151
## Recency..months.:Time..months.      0.0018254  0.0008480   2.153 0.031351
## Frequency..times.:Time..months.    -0.0013911  0.0007555  -1.841 0.065573
##                                       
## (Intercept)                        .  
## Recency..months.                   ***
## Frequency..times.                  ***
## Time..months.                      ** 
## Recency..months.:Frequency..times. .  
## Recency..months.:Time..months.     *  
## Frequency..times.:Time..months.    .  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 553.37  on 499  degrees of freedom
## Residual deviance: 467.96  on 493  degrees of freedom
## AIC: 481.96
## 
## Number of Fisher Scoring iterations: 5
```

```r
predictor = predict.glm(interstep,newdata=test)
interpredict = cbind(predictor,test[,5])
```

```r
a=ROC(interpredict)
```

![Performance of logistic regression predictor with interaction effects](figure/unnamed-chunk-15-1.png) 
So... that's actually the best prediction (Figure 5), if you give it points for less parameters (kinda), but it's still nothing to write home about.  Probably the interaction is meaningful, so it's helping a little, but we remain unamused by the performance of this predictor.  

In various runs, the logistic-with-interactions precision seems generally more reliable than any other predictor, but I haven't strictly quantified it with cross-validation. Specifically, the first few predictions here seem to be more accurate than the others, and precision takes longer to decay.

### Addendum
After doing this analysis, I looked at the paper from which the dataset came.  They made things more complicated by treating the blood donation visits as a series of Bernoulli trials.  Maybe they had more complete data than we, but it seemed to me that without knowing the starting point or the number of actual blood drives these people had gone through it was weird to model it that way.  And if I interpret their performance properly, their model is no better or a little worse than the ones I present.

Given that there are really only 3 independent variables for prediction ("Monetary" is just a linear transformation of "Frequency"), this is pretty squarely a high-n low-p problem. My intuition is that scraping for slightly better performance with more complicated methods is more likely to cause trouble and mislead people through overfitting than to add any additional power.  With this somewhat skeptical view of how data analysis is generally performed, I rest my case.

## Addendum 2: Nearest neighbor.  
Apparently nearest-neighbor is good. I am trying it out.  Figure 6: k=2, Figure 7: k=3, Figure 8: k=4, Figure 9: k=5.

```r
library(class)
nn2_pred = knn(train[,1:4],test=test[,1:4] ,cl=train[,5],k=2)
nn2_predict = cbind(nn2_pred,test[,5])
```

```r
a=ROC(nn2_predict)
```

![Performance of kNN with k=2.](figure/unnamed-chunk-17-1.png) 

```r
nn3_pred = knn(train[,1:4],test=test[,1:4] ,cl=train[,5],k=3)
nn3_predict = cbind(nn3_pred,test[,5])
a=ROC(nn3_predict)
```

![Performance of kNN with k=3.](figure/unnamed-chunk-18-1.png) 

```r
nn4_pred = knn(train[,1:4],cl=train[,5],test=test[,1:4] ,k=4)
nn4_predict = cbind(nn4_pred,test[,5])
a=ROC(nn4_predict)
```

![Performance of kNN with k=4.](figure/unnamed-chunk-19-1.png) 

```r
nn5_pred = knn(train[,1:4],test[,1:4],cl=train[,5] ,k=5)
nn5_predict = cbind(nn5_pred,test[,5])
a=ROC(nn5_predict)
```

![Performance of kNN with k=5.](figure/unnamed-chunk-20-1.png) 
