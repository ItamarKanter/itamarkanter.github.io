---
layout: post
mathjax: true
comments: true
title: "AUC- Benchmarking"
date: 2019-06-04
categories: blogging
tags: R AUC benchmark
---
In the previous blog post we have shown that AUC is merely a measure of
order. From such point of view it’s intuitive to expand the definition
of AUC also to regression problems where $$y\in \mathbb{R}$$ is a real value and not
just a binary output. I leave to the reader to test this approach as an
evaluation metric for regression problems. Currently, there are several
R packages provided functions for calculating AUC
([AUC](https://cran.r-project.org/web/packages/AUC/index.html),
[pROC](https://cran.r-project.org/web/packages/pROC/index.html),
[mltools](https://cran.r-project.org/web/packages/mltools/index.html),
[ROCR](https://cran.r-project.org/web/packages/ROCR/)).

``` r
set.seed(2019)
n <- 3e3
y_pred <- runif(n)
y_true  <- sample(c(TRUE, FALSE), n, replace = TRUE)
# y_true  <- sapply(y_pred, rbinom, size=1, n=1)
AUC_score <- vector(mode="list", length=4)
names(AUC_score) <- c("AUC", "pROC", "mltools", "ROCR") 
AUC_score["AUC"] <- AUC::auc(AUC::roc(y_pred, as.factor(as.numeric(y_true))))
AUC_score["pROC"] <-  pROC::auc(y_true, y_pred, direction = "<")
AUC_score["mltools"] <-  mltools::auc_roc(y_pred, y_true)
AUC_score["ROCR"] <-  ROCR::performance(ROCR::prediction(y_pred, y_true), "auc")@y.values
print(AUC_score)
```

    ## $AUC
    ## [1] 0.5029898
    ## 
    ## $pROC
    ## [1] 0.5029898
    ## 
    ## $mltools
    ## [1] 0.5029898
    ## 
    ## $ROCR
    ## [1] 0.5029898

OK, The outputs are all equal. Now we will benchmark the calculation
time. In order to have a reference I also include in the comparison two
methods for measuring order (i.e. combination of `rank` and `order`, and
fast variants of *[Kendall
correlation](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient))*

``` r
benchmarl_res <- microbenchmark::microbenchmark(list = alist(
  AUC = AUC::auc(AUC::roc(y_pred, as.factor(as.numeric(y_true)))),
  pROC = pROC::auc(y_true, y_pred, direction = "<"),
  mltools = mltools::auc_roc(y_pred, y_true),
  ROCR = ROCR::performance(ROCR::prediction(y_pred, y_true), "auc")@y.values,
  sort = rank(y_true[order(y_pred)]),
  kendall_fast = pcaPP::cor.fk(y_pred,y_true)
), times=50
)
print(benchmarl_res)
```

    ## Unit: microseconds
    ##          expr       min        lq       mean    median        uq       max
    ##           AUC 16781.517 17327.742 18518.4937 17937.638 18997.834 29182.564
    ##          pROC  7952.405  8142.260  9100.5361  8735.203  9704.918 12823.513
    ##       mltools  6599.770  6803.761  7466.5183  7090.685  7904.106 11205.881
    ##          ROCR  2510.748  2694.543  2874.4553  2799.450  2987.015  4012.704
    ##          sort   447.574   463.493   497.5280   489.325   516.231   689.321
    ##  kendall_fast   285.995   310.154   671.9544   340.054   372.969 16345.131
    ##  neval
    ##     50
    ##     50
    ##     50
    ##     50
    ##     50
    ##     50

``` r
require("ggplot2")
autoplot(benchmarl_res)
```

![](/images/AUC_benchmark_files/figure-markdown_github/unnamed-chunk-2-1.png)

Pretty slow, maybe we can use the swap based inversion counting to make
it faster.

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

AUC_swap_based_n2<-function(y_pred, y_true){
  n_of_swaps = count_swaps(y_true[order(y_pred)])
  maximal_swaps<-max_swaps(y_true)
  AUC = 1-n_of_swaps/maximal_swaps
  return(AUC)
}
```

``` r
benchmarl_res <- microbenchmark::microbenchmark(list = alist(
  AUC_swap_naive = AUC_swap_based_n2(y_pred, y_true)), times=5)
print(benchmarl_res)
```

    ## Unit: seconds
    ##            expr      min       lq     mean   median       uq      max
    ##  AUC_swap_naive 1.055978 1.056846 1.070697 1.056912 1.091747 1.092003
    ##  neval
    ##      5

``` r
print(paste0('AUC=', AUC_swap_based_n2(y_pred, y_true)))
```

    ## [1] "AUC=0.502989777777778"

More than a second! there is still a room for improvement. Now, we will
turn into $$O(nlog(n))$$ swap counting algorithm (instead of $$O(n^2)$$ for the naive algorithm). The algorithm is based on
the [merge sort algorithm](https://en.wikipedia.org/wiki/Merge_sort) and
will be implemented in C++.

``` r
library(Rcpp)
src <- "
//from https://www.geeksforgeeks.org/counting-inversions/
//and http://macherkime.blogspot.com/2016/02/merge-sort-rcpp-integration-code.html
/*counting inversions function using the merge sort algorithm
*/

#include <Rcpp.h>
using namespace Rcpp;

int intercal(int p, int q, int r, NumericVector v,  NumericVector w)
{
  int i, j, k, inv_count;
  i = p;
  j = q;
  k = 0;
  inv_count = 0; 

  while (i < q && j < r) {
  if (v[i] <= v[j]) {
  w[k] = v[i];
  i++;
  }
  else {
  w[k] = v[j];
  j++;
  inv_count = inv_count + q - i;
  }
  k++;
  }
  while (i < q) {
  w[k] = v[i];
  i++;
  k++;
  }
  while (j < r) {
  w[k] = v[j];
  j++;
  k++;
  }
  for (i = p; i < r; i++)
  v[i] = w[i-p];
  return inv_count;
}

int mergesort(int p, int r, NumericVector v, NumericVector aux)
{
  int q, inv_count = 0; ;
  if (p < r - 1) {
  q = (p + r) / 2;
  inv_count = mergesort(p, q, v,aux);
  inv_count += mergesort(q, r, v,aux);
  inv_count += intercal(p, q, r, v,aux);
  }
  return inv_count; 
}

// [[Rcpp::export]]
int fast_inv_count(NumericVector vetor) {
Rcpp::NumericVector res = Rcpp::clone(vetor);
Rcpp::NumericVector aux = Rcpp::clone(vetor);
int n = res.size();
int inv_count;
inv_count = mergesort(0,n,res,aux);
return inv_count;}"
sourceCpp(code = src)

AUC_swap_based_nlogn<-function(y_pred, y_true){
  n_of_swaps = fast_inv_count(y_true[order(y_pred)])
  maximal_swaps<-max_swaps(y_true)
  AUC = 1-n_of_swaps/maximal_swaps
  return(AUC)
}
print(paste0('AUC=', AUC_swap_based_nlogn(y_pred, y_true)))
```

    ## [1] "AUC=0.502989777777778"

``` r
benchmarl_res <- microbenchmark::microbenchmark(list = alist(
  AUC_swap_based_nlogn = AUC_swap_based_nlogn(y_pred, y_true)), times=50)
print(benchmarl_res)
```

    ## Unit: milliseconds
    ##                  expr      min       lq     mean   median       uq
    ##  AUC_swap_based_nlogn 1.246858 1.313925 1.696874 1.434768 1.685148
    ##       max neval
    ##  5.724574    50

Around 1.7 ms, much better than the naive approach.

Recap
=====

A combination of proper algorithmic (merge sort) and efficient
implementation in C++ provide us fast implementation of AUC calculation.
I hope that in this blog-post I convey the concept of what is AUC (hint:
measure of order) and how to find it in a quick manner.
