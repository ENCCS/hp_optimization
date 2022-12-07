Model selection
===============

Cross validation is an important technique for 
getting insights into the statistical robustness of our models. 
Essentially, we perform many *resamples* of our dataset, each resample 
having different data points in its training/test set split. This 
means that we're able to estimate how much the performance of our 
methods vary with different choices of training and test data.

Unfortunately, in the neural networks litterature, it's seldomly 
used when presenting new models. The reason is mainly two-fold: 
chasing performance on *canonical* public test sets and the time of 
running experiments.

Neural Networks in the machine learning litterature are often 
trained on large datasets (the settings where they are competative) 
where some earlier work created a *canonical* split of the dataset 
into at least a training and test set, but sometimes even a 
development (validation) set. To make your new model comparable to 
the previous, you need to use at least the same test set, which 
discourage you to also perform cross validation to estimate 
generalization error.

If you are training models to mainly publish State-of-the-art 
results on some well known benchmark, you are more or less forced 
into using performance on the canonical test set.

If on the other hand you are developing models for use in practice 
you definitely should be using some form of resampling.


Cross validation for neural networks
------------------------------------





Measuring the effect of hyper parameters
----------------------------------------

