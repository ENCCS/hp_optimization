================================================
Hyper Parameter optimization for neural networks
================================================

Many machine learning algorithms can be formulated as methods of trying to find some 
parameters :math:`\mathbf{\theta}` which makes a parameterized function :math:`f(x; \mathbf{\theta})`
approximate some true unknown relationship between two variables :math:`x` and :math:`y`.

The family of functions which we can span with :math:`\mathbf{\theta}` can 
be thought of as a set. Each specific setting of the parameters is an element 
of this set. The *learning* we perform when optimizing a Neural Network 
is searching for good points in this set of functions. We use some search 
procedure such as local gradient search to explore this set using our training data.

What is fundamentally important when designing Neural Networks is the extent of 
this set: what functions we can actually end up in. This is determined by the 
modeling choices we make, such as the size and kinds of layers we use.

This set is determined by how we *design* our Neural Network. We can think of the 
hyper parameters we choose as defining this set of functions. In this workshop we 
will look at methods to tune these settings automatically using the Optuna framework.

Hyper Parameters in Neural Networks
-----------------------------------
In Neural Network, many parts of the function family can be determined by hyper parameters. 
The number of layers, the size of layers, the activation functions, dropout rate, 
learning rate, skip-connections, normalization layers, mini-batch size etc. 

Often we focus on just a couple of these parameters, since the difficulty of any search 
method in hyper parameter space tends to increase exponentially with the number of 
hyper parameters. One hyper parameter in particular has always been important to optimize 
for neural networks trained with gradient descent: the *learning rate*.


The pesky learning rate
-----------------------

It's also long been known that the *learning rate* is central to 
how well the SGD search will be perform when training neural networks, 
set it too low and the optimization doesn't converge, set it too high and it diverges.

It's also been noted that the learning rate often benefits from *scheduling*, i.e. 
changing it over the course of the optimization. These schedules can vary a lot, but are 
mostly parameterized by some additional set of hyper parameters, such as how much and 
how often to reduce the learning rate.

There are modern automatic learning rate finders which perform a greedy local search by scanning 
a range of learning rates over a number of training batches (popularized as :code:`lr_range_finder` 
by fast.ai and incorporated in many machine learning frameworks). These have proven successful and could be 
attempted instead of what we're doing in this workshop which uses a more general approach.

In addition to tuning a learning rate, we often use variations of SGD with features such 
as momentum and scaling the update step by some exponentially moving average of the gradients. 
These mechanism are typically also parameterized (by *momentum* parameters, such as :math:`\beta_1` 
and :math:`\beta_2` of the Adam optimizer). 

It's also common to use *weight decay*, which pushes weights towards zero, and the strenght 
of this mechanism has to be determined.

Dropout
-------

A common method to make neural networks less prone to overfitting is that of dropout, 
essentally a way of faking ensamble models within a single neural network. This is also 
governed by a hyper parameter which determines how large subnetworks to sample from 
the original network.

Architecture
------------

Size of layers and number of layers also greatly affects how the neural network 
will perform. These are fundamentally 



The sensitivity to hyper parameters
-----------------------------------

Unfortunately for neural networks, they are very sensitive to hyper parameter choice. 
In particular learning rate has been shown to dramatically affect performance. This 
means that any study which compares different neural networks (with different architectures 
or datasets) is meaningless unless all have undergone their own hyper parameter 
optimization procedure. 
This is also true for when we experiment with our own architectures: if we just run 
the SGD procedure with some arbitrary hyper parameter point (e.g. a learning 
rate of :math:`0.001`) that doesn't necessarily reflect the best 
performance of any single architecture and the choice whe eventually make will be 
biased from the choice of "default" hyper parameters.


Can I make HP search faster by using a subset of the data?
----------------------------------------------------------

While you can certainly make it faster, the hyper parameters you find will 
depend on the dataset. Even if you don't search over architectural parameters 
(e.g. size of layers, connectivity between layers) many hyper parameters are 
still tied to effective model complexity (e.g. learning rate, weight decay, 
dropout rate) which means that the ones your HP search procedure finds are 
likely conservative if you increase the dataset size.


Can I make HP search faster by training for fewer epochs?
---------------------------------------------------------

Another way of speeding up HP search could be to train for 
just a few number of epochs and use early performance as a 
proxy for late performance. The idea is that networks where 
the error decreases more rapidly early should be better 
than those where it decreases more slowly.

Unfortunately, studies have found that early performance doesn't 
correlate significanly with late performance. The training of networks
can show "plateaus", where loss stops decreasing, but might then 
start decreasing again. In particular, turning on or off *Batch Normalization* 
(BN) shows this performance, where BN typically takes a longer time to start 
decreasing, but in the end converges to models which outperform non-BN ones.

This means that in principal we should train our networks to convergence 
before determining how suitable a set of hyper parameters were.

There are situations where this is infeasible, models being trained 
on truly massive datasets (e.g. large scale language models trained 
on text scraped from the web) are never trained to convergence. 
Their development error continually decreases until a new run is started. 
In practice, it's very rare that we have these kinds of datasets. 
If you do, many of these recommendations are not valid.  



Hyper Parameter optimization and experiment design
--------------------------------------------------

The field of experiment design (or design of experiments, DOE) has a 
long and solid history of building methods to determine what experiments 
to perform to optimize some property and is fundamentally important to 
all of experimental science.
Hyper parameter optimizaition is really just another application of 
experiment design.
HP optimzation for neural nets often 
have a lot of factors (hyper parameters) so methods such as 
factorial design becomes intractable. Another complication is that 
many hyper parameters have higher order interactions, i.e. the 
outcame changes by factors in combination and not linearly. Also, 
the levels of the factors (the values to investigate for the hyper paramters) 
are often large, so a two-level factorial design is insufficient.

The field of HP optimization has also been decoupled from DOE, which leads to 
it developing its own tools for doing the optimization. It's important 
to note though, that any expertise you have in DOE can 
inform you of how to do HP optimzation and *vice versa*. 
The tools we use in this workshop to optimize hyper parameters could 
easily be used to optimize any experimental factors.

One important method in experiment design is that of the Response Surface Method (RSM). 
Essentially, we can fit some model you our experiment settings and their outcomes. 
We can then probe this model (the response surface) for settings which should be 
optimal and iteratively rebuild the model. 
Classical RSM uses a second degree polynomial for this purpose, since it's easy to use for
optimzation and is able to capture second order interactions. However, this 
model might suffer when modelling the hyper parameter respense for neural networks, in 
particular because we often have some categorical factors.

This has led to the development of more flexible models for the response surface, and one 
popular such model has been Gaussian Processes. This model has the advantage of also modeling 
the uncertainty over the response surface which allows the hyper parameter optimization procedure 
to select points which gives as much information as possible. See `The Supervised Machine Learning Book <http://smlbook.org/GP>`_ for an excellent illustration.

Unfortunately, Gaussian Processes suffer when the factors are discrete, which they often are 
in Neural Network hyper parameters. The default sampling procedure for Optuna is a method called 
Tree-Structured Parzen Estimators.