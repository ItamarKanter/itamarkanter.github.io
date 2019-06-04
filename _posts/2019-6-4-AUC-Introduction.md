---
layout: post
mathjax: true
comments: true
title: "AUC- Introduction"
date: 2019-06-04
categories: blogging
tags: R AUC
---
AUC (Area Under the Curve) is a common name to depict the area under a
function. Specifically in the fields of statistics and machine learning
this term refer to the area under the Receiver Operator Characteristic
(ROC), which is the curve defined by the pairs (FPR(t), TPR(t)) at
different threshold (t) levels. Where FPR stands for False Positive
Rate, and TPR for True Positive Rate. $FPR=\\frac{FP}{FP+TN}$ and
$TPR=\\frac{TP}{TP+FN}$,

TP- True positive, The model predicted positive, the example was
positive

FP- False positive, The model predicted positive, the example was
negative

TN- True negative, The model predicted negative, the example was
negative

FN- False negative, The model predicted negative, the example was
positive

So, the TPR is the fraction of correct positive predictions out of all
the positive examples (also known as recall), and the FPR is the
fraction of incorrect positive prediction out of all negative examples.
The ROC is monotonically increase function and constrained in
*T**P**R*, *F**P**R* ∈ \[0, 1\]. Let’s dive in with a simple example we
will create in R.

``` r
library("dplyr")
y_pred <- c(0.2, 0.42, 0.46, 0.55, 0.6, 0.66, 0.8, 0.9, 0.92, 0.95)
y_true <- c(0, 0, 1, 0, 1, 0, 1, 1, 1, 1 )
df <- data.frame(y_pred, y_true)
print(df)
```

    ##    y_pred y_true
    ## 1    0.20      0
    ## 2    0.42      0
    ## 3    0.46      1
    ## 4    0.55      0
    ## 5    0.60      1
    ## 6    0.66      0
    ## 7    0.80      1
    ## 8    0.90      1
    ## 9    0.92      1
    ## 10   0.95      1

Let’s find the FPR(t), TPR(t) at *t* = 0.5

``` r
t = 0.5
df <- df %>% rowwise() %>% mutate(y_pred_out = as.numeric(y_pred>t))
print(df)
```

    ## Source: local data frame [10 x 3]
    ## Groups: <by row>
    ## 
    ## # A tibble: 10 x 3
    ##    y_pred y_true y_pred_out
    ##     <dbl>  <dbl>      <dbl>
    ##  1   0.2       0          0
    ##  2   0.42      0          0
    ##  3   0.46      1          0
    ##  4   0.55      0          1
    ##  5   0.6       1          1
    ##  6   0.66      0          1
    ##  7   0.8       1          1
    ##  8   0.9       1          1
    ##  9   0.92      1          1
    ## 10   0.95      1          1

``` r
TP <- sum(as.logical(df$y_true) & as.logical(df$y_pred_out))
FP <- sum(!as.logical(df$y_true) & as.logical(df$y_pred_out))
TN <- sum(!as.logical(df$y_true) & !as.logical(df$y_pred_out))
FN <- sum(as.logical(df$y_true) & !as.logical(df$y_pred_out))
TPR <- TP/(TP+FN)
FPR <- FP/(TP+FP)
print(TPR)
```

    ## [1] 0.8333333

``` r
print(FPR)
```

    ## [1] 0.2857143

However changing the threshold to *t* = 0.51 or *t* = 0.49 won’t change
the results, so it make sense to calculate the TPR(t), FPR(t) for
*t* ∈ {0, *y*<sub>*p**r**e**d*</sub>} (I add 0 to include the extreme
point of the curve). We can implement this in a simple function and a
plot the ROC

``` r
fpr_tpr_t <- function(y_pred, y_true){
  y_true <- as.logical(y_true)
  condition_true <- sum(y_true)
  condition_false <- length(y_true) - sum(y_true)
  t <- unique(c(0, y_pred))
  tpr <- fpr <- numeric(length(t))
  for (i in seq_along(t)){
    threshold <- t[i]
    tpr[i] <- sum(y_pred>threshold & y_true) / condition_true
    fpr[i] <- sum(y_pred>threshold & !y_true) / condition_false
  }
  return (list(t=t, tpr=tpr, fpr=fpr))
}
res = fpr_tpr_t(df$y_pred, df$y_true)

plot(res$fpr,res$tpr,xlab="FPR", ylab="TPR", type='b')
abline(a=0, b=1, lty=2)
```

![](/images/AUC_files/figure-markdown_github/unnamed-chunk-3-1.png) 
The diagonal line represents the Null model, i.e. a random prediction of y\_pred.

How can we find the AUC? Note that the size of horizontal and vertical
steps between adjacent point is equal. The vertical step size is simply
$\\frac{1}{\\sum{I(y\_{true}=1)}}$ and the horizontal step size is
$\\frac{1}{\\sum{I(y\_{true}=0)}}$. In our toy example the AUC is :

``` r
1-(1/sum(as.logical(df$y_true))*1/sum(!as.logical(df$y_true))*3)
```

    ## [1] 0.875

We can validate the results with the `AUC` package

``` r
require(AUC)
auc(roc(df$y_pred,as.factor(df$y_true)))
```

    ## [1] 0.875

AUC from different perspectives
===============================

Until now we introduce the AUC score from a traditional perspective that
you can find in any machine learning/statistics text book. In the next
sections I would like to describe how we can view the meaning of the AUC
from additional perspectives.

AUC and the GINI score
----------------------

The [Gini score](https://en.wikipedia.org/wiki/Gini_coefficient) is a
measure statistical dispersion, intend to measure the inequality in
wealth or income of a population. The Gini coefficient is determine by
plotting the Lorenz curve, which is graph of the cumulative share of
persons vs. the cumulative share of wealth. In our case, you can switch
persons with samples, and wealth with positive predictions (*y* = 1).

``` r
lorenz_curve <- function(pred, y){
  cum_y <- c(0, cumsum(y[order(pred)])/sum(y))
  cum_samples <- seq(0, 1, length.out=length(cum_y))
  return (list(cum_y=cum_y, cum_samples=cum_samples))
}
res_pred <- lorenz_curve(df$y_pred, df$y_true)
plot(res_pred$cum_samples, res_pred$cum_y, xlab="cumulative #samples", ylab="cumulative #predictions", type='b', ylim=c(0, 1), xlim=c(0, 1))
abline(a=0, b=1, lty=2)
polygon(c(res_pred$cum_samples), c(res_pred$cum_y), col = rgb(1, 0, 0,0.5) )
polygon(c(res_pred$cum_samples, 1), c(res_pred$cum_y, 0), col = rgb(0, 1, 0,0.5) )
text(0.5, 0.42, "A", col='white', cex=2)
text(0.8, 0.22, "B", col='white', cex=2)
legend("topleft",
  legend = c("equality line", "Lorenz curve"),
  lty =c(2,1)
)
```

![](/images/AUC_files/figure-markdown_github/unnamed-chunk-6-1.png) The Gini
coefficient is the the ratio $Gini=\\frac{A}{A+B}=2A$ (because
*A* + *B* = 0.5)

``` r
midPoints <- function(x){
  (x[-length(x)]+x[-1])/2
}
B = sum(midPoints(res_pred$cum_y) * diff(res_pred$cum_samples))
A = 0.5 - B
Gini = 2*A
print(Gini)
```

    ## [1] 0.3

Now we will find the Gini coefficient with respect to the perfect model
(i.e. the true labels that maximize the inequality)

``` r
res_pred <- lorenz_curve(df$y_pred, df$y_true)
res_true <- lorenz_curve(df$y_true, df$y_true)
plot(res_pred$cum_samples, res_pred$cum_y, xlab="cumulative #samples", ylab="cumulative #predictions", type='b', ylim=c(0, 1), xlim=c(0, 1), lty=1)
lines(res_true$cum_samples, res_true$cum_y, lty=3, lwd=3)

abline(a=0, b=1, lty=2)
polygon(c(res_pred$cum_samples), c(res_pred$cum_y), col = rgb(1, 0, 0,0.5) )
polygon(c(res_pred$cum_samples, rev(res_true$cum_samples)), c(res_pred$cum_y, rev(res_true$cum_y)),
        col = rgb(0, 1, 0,0.5) )
text(0.5, 0.42, "A", col='white', cex=2)
text(0.35, 0.1, "B", col='white', cex=2)
legend("topleft",
  legend = c("equality line", "Lorenz curve predictions", "Lorenz curve labels"),
  lty =c(2,1,3), lwd=c(1,1,3)
)
```

![](/images/AUC_files/figure-markdown_github/unnamed-chunk-8-1.png) The relative
Gini coefficient is:

``` r
area_between_curves <- function(x, f1, f2){
  mid_points = abs(f1-f2)
  mid_height = (mid_points[-1] + head(mid_points, -1))/2
  area = sum(mid_height*diff(x))
  return(area)
}
equality_line = res_pred$cum_samples
S_A = area_between_curves(res_pred$cum_samples, equality_line, res_pred$cum_y)
S_B = area_between_curves(res_true$cum_samples, res_true$cum_y, res_pred$cum_y)
Gini = S_A/(S_A+S_B)
print(Gini)
```

    ## [1] 0.75

And we can find the AUC from the Gini coefficient by
$AUC=\\frac{Gini+1}{2}$

``` r
AUC = (Gini+1)/2
print(AUC)
```

    ## [1] 0.875

AUC and the Wilcoxon rank-sum test
----------------------------------

[Wilcoxon rank-sum
test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test) is a
non parametric statistic test. It’s check whether a randomly selected
sample from one set will be greater than randomly selected sample from a
second set. To obtain the Mann-Whitney U statistic we simply aggregate
the two sets and rank them. The U statistics is defined to be
$U\_1=R\_1 - \\frac{n\_1(n\_1+1)}{2}$, where *R*<sub>1</sub> is the sum
of ranks of samples from the first set.

``` r
U_stat <- function(labels, score){
  R1 = sum(order(score)[labels==1])
  n1 = sum(labels==1)
  U1 = R1 - n1*(n1+1)/2
  return(U1)
}
print(U_stat(df$y_true, df$y_pred))
```

    ## [1] 21

And the relation to AUC is $AUC=\\frac{U\_1}{n\_1n\_2}$

``` r
n1 = sum(df$y_true==1)
n2 = sum(df$y_true==0)
AUC = U_stat(df$y_true, df$y_pred)/(n1*n2)
print(AUC)
```

    ## [1] 0.875

AUC and the Kendall rank correlation coefficient (or how many swaps we need to order our predictions to fit the true outcome)
-----------------------------------------------------------------------------------------------------------------------------

Last but not least, the elegant way (in my opinion) to look on the AUC
score is as distance in terms of number “bubble sort” swaps (swaps of
adjacent elements in the prediction vector, normalized by the maximal
number of swaps) required to reach perfect prediction.

``` r
count_swaps<-function(x){
  n_swaps=0
  n<-length(x)
  for(j in 1:(n-1)){
    for(i in 1:(n-j)){
      if(x[i]>x[i+1]){
        temp<-x[i]
        x[i]<-x[i+1]
        x[i+1]<-temp
        n_swaps = n_swaps+1
      }
    }
  }
  return (n_swaps)
}

max_swaps<-function(y_true){
  max_swaps<-length(y_true)*(length(y_true)-1)/2
  #correcting for identical elements
  for (i in table(y_true)){
    if (i>1){
      max_swaps<-max_swaps-(i*(i-1)/2)
    }
  }
  return(max_swaps)
}

n_of_swaps = count_swaps(df$y_true[order(df$y_pred)])
maximal_swaps<-max_swaps(df$y_true)
AUC = 1-n_of_swaps/maximal_swaps
print(AUC)
```

    ## [1] 0.875
