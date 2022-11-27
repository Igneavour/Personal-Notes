---
tags: [constraint, clustering, unsupervised-learning, measure-of-similarity, dimensionality-reduction, convergence, principal-component-analysis, eigenanalysis]
aliases: [MLNN T5, Machine Learning and Neural Networks Topic 5]
---

# Reading resources for this topic

1. [Chapter 6.1 and 6.3 - dimensionality reduction and PCA](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=3339851)
2. [Chapter 7.1 - 7.3 - Clustering (k-means)](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=3339851)

# Unsupervised Learning

- no labels, therefore need some other constraint
- choice of constraint is depedent on task 
	- with input observations placed, outputs should satisfy constraints

## Measure of similarity

- constraints can be some measure of similarity where we map observations that are similar in X to outputs are close together in Y
	- this is an example of clustering

## Dimensionality reduction

- constraints can also be finding ways to reducing the size of X, the dimensionality of X while maximizing the variance between the data points once we get to mapping in Y

# K-means

![[k_mean_1.png]]

1. Initialise k random centroids

![[k_mean_2.png]]

2. Calculate distance from every data point to centroid

![[k_mean_3.png]]

3. Assign data points to nearest centroid to form clusters (K equals number of clusters to form and hence 2 for this case)

![[k_mean_4.png]]

4. Re-assign centroids as the mean of each cluster's data

![[k_mean_5.png]]

5. Repeat steps 2-4 until convergence (centroids stop changing)

### Convergence in k-means subject to local minima

- the location of initial values will affect the final outcome of where the final clusters will be
- we need to choose good initial centroids
- restart several times with different centroids

### Scaling data is also important for k-mean

- can perform min-max scaling (scale between 0-1)

# Dimensionality reduction

![[dimensionality_reduction_features.png]]

## Principal Component Analysis (PCA)

![[PCA_1.png]]

1. Find the place with the most variability in that 3D space, and draw a red line where there is the most deviation from the mean, which is the higher variance

![[PCA_2.png]]

2. We can then find the next direction which is orthogonal to that red line (90 deg), which has the next most chenge in the data

![[PCA_3.png]]

3. By taking that two principal components and rotating it, we get this 2D with the PC as the new axes, we get a space in 2D where the first PC is the x-axis and the second PC is the y-axis

![[PCA_4.png]]

4. We can also further reduce the dimensionality to 1D as seen in the above diagram

### Mathematical expression

Given data X with F features and N samples:

$$ X=[x^1x^2,...x^N] \text{, where X} \epsilon R^{FxN} $$
The sample mean is then:

$$ \bar X = \frac{1}{N}\sum_{i=1}^N x^i \text{,  where } x^i \epsilon R^{Fx1} $$
The covariance is:

$$ cov(X) = S = \frac{1}{N} \sum_{i=1}^N (x^i - \bar x)(x^i - \bar x)^T $$

- covariance captures how the N variables vary together - how they change together with respect to the mean

![[PCA_transformation.png]]

- transforming our 3D input into 2D
- to get W, we need covariance of output as big as possible

![[finding_W_in_PCA.png]]

- I in W^TW = I is the identity matrix
- find out  more about eigenanalysis in further readings

![[PCQ_with_eigenanalysis.png]]

- output is Y = W2 transpose product with X

### Data compression

![[data_compression_PCA.png]]

- to recover principal components given matrix X, we use eigenanalysis on the covariance matrix of our data

![[PCA_mini_quiz.png]]

- just need a single dimension to represent this data