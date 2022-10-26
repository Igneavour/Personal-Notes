---
tags: [linear, regression, mean-squared-error, 1D-gradient-descent, 2D-gradient-descent, learning-rate, partial-derivative, stochastic-gradient-descent, mini-batch, multivariate-linear-model, data-scaling, min-max-normalisation, range-normalisation, standardization]
aliases: [MLNN T3, Machine Learning and Neural Networks Topic 3]
---

# Reading resources for this topic

1. [Cheatsheet for linear algebra](http://ml-cheatsheet.readthedocs.io/en/latest/linear_algebra.html)
2. [Scikit-learn datasets](http://scikit-learn.org/stable/datasets/index.html)
3. [Further datasets](http://scikit-learn.org/stable/datasets/index.html#downloading-datasets-from-the-mldata-org-repository)
4. [Short tutorial on using linear algebra and matrices in Python](http://ml-cheatsheet.readthedocs.io/en/latest/linear_algebra.html)
5. [Chapter 2 section 2.4 gradient descent](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=6642860)

# Linear Regression

Hypothesis:
$$ h_{\theta}(x) = \theta_0 + \theta_1x_1 $$
$$ = \sum_{j=0}^1\theta_jx_j \text{ (with }x_0 = 1) $$
$$ = [1 \text{ }x]\left[\begin{array}{cc}\theta_0\\\theta_1\end{array}\right] $$
$$ \text{where }\theta_0 \text{ is the y-intercept and} $$
$$ \theta_1 \text{ is the gradient} $$

![[linear_regression_example_1.png]]

## Using data to learn hypothesis

### Example from 3.102

![[linear_regression_example_2.png]]

The red arrows in the example above is the L2 loss, or "Mean Squared Error":

$$ J(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^i)-y^i)^2 $$

# Gradient Descent in 1D

- to find the optimal parameter values that helps to minimise the loss OR
$$ \text{Find the gradient, }\theta_1\text{, that minimises loss, }J(\theta) $$

![[L2_loss_gradient_descent.png]]

- From the graph, we can see that as the gradient is increased, the loss decreases and eventually reaches a global minimum of 0
- Plotting those loss will form a convex curve as you proceed on with increasing the gradient

## Differentiating the Loss

$$ J(\theta) = \frac{1}{2m}\sum_{i=1}^m(\theta_0+\theta_1 x^i-y^i)^2 $$

Using partial derivatives, 
$$ J_1'(\theta) = \frac{\delta J(\theta)}{\delta \theta_1} = \frac{1}{m}\sum_{i=1}^m(\theta_0 + \theta_1x_1^i-y^i) * x_1^i  $$

## Gradient descent update rule

$$ \theta_1^2 = \theta_1^1 - \alpha J_1'(\theta^1) $$

- the alpha value is known as the <b>convergence rate</b>
	- it changes the speed at which your gradient descent algorithm will happen

## Learning Rate (alpha)

- usually set between 0 and 1
- if its too small, the gradient descent algorithm moves too slowly and takes very long to converge vice-versa

$$ \theta_1^{new} = \theta_1^{old} - \alpha J'(\theta^{old}) $$

# Gradient Descent in 2D (batch gradient descent)

- In 2D, we need to also think about the partial derivative with respect to theta 0

## Gradient

$$ J_0'(\theta) = \frac{\delta J(\theta)}{\delta \theta_0} = \frac{1}{m}\sum_{i=1}^m(\theta_0 + \theta_1x_1^i-y^i)  $$
$$ J_1'(\theta) = \frac{\delta J(\theta)}{\delta \theta_1} = \frac{1}{m}\sum_{i=1}^m(\theta_0 + \theta_1x_1^i-y^i) * x_1^i  $$

![[gradient_descent_2D_example_1.png]]

<p style="font-size:30px">Slow for lots of data</p>

# Stochastic Gradient Descent

- Fast for lots of data
- Doesn't converge smoothly as it chooses one of the data points at random and we do our update based on that

# Mini batch

- hybrid solution of stochastic and normal gradient descent
- does not take all data points but take a random sub selection of data

# Multivariate Linear Model

$$ h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n $$
$$ = \sum_{j=1}^n\theta_jx_j+\theta_0 $$
$$ = \sum_{j=0}^n\theta_jx_j \text{ (with }x_0=1) $$
$$ = \theta^Tx \text{ (with }x_0=1) $$

## Data Scaling

- theta values can be very different from one another due to data values being very different in scale from one another
- to solve this issue, we use feature scaling such as min-max normalisation (between 0 and 1)

### Min-Max normalisation

$$ x_j^S = \frac{x_j - min(x_j)}{max(x_j) - min(x_j)} $$

### Range normalisation (centred on mean)

- works well if data tends to be normally distributed

$$ x_j^S = \frac{x_j - mean(x_j)}{max(x_j) - min(x_j)} $$

### Standardization (z-score)

$$ x_j^S = \frac{x_j - mean(x_j)}{std(x_j)} $$

# Polynomial regression

![[polynomial_regression.png]]

- non-linear in x
	- BUT linear in parameter space
	- gradient descent can be used