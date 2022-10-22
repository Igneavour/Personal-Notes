---
tags: [supervised-learning, k-nearest-neighbour-classification, euclidean-distance, manhattan-distance, label, decision-tree, decision-boundary, classification-and-regression-tree, classifier-evaluation, confusion-matrix]
aliases: [MLNN T2, Machine Learning and Neural Networks Topic 2]
---

# Reading resources for this topic

1. [nearest neighbour module documentation for scikit-learn](http://scikit-learn.org/stable/modules/neighbors.html)
2. [decision trees tutorial for scikit-learn](https://scikit-learn.org/stable/modules/tree.html)
3. [chapter 1, section 1.2 about decision tree classifiers](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=6642860&ppg=1)
4. [section 2.1 and 2.5 provide a good overview of supervised classification](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=3339851&query=introduction+to+machine+learning+alpaydin)

# Supervised Learning

## k-Nearest Neighbour Classification (k-NN)

- A lazy-learning algorithm with 2 main parameters. 
	- K, which is the number of nearest neighbours that we are going to look for
	- Similarity or distance measure, which allows us to compare the different data points

### Detect test dataset

- Measure <b>distance</b> from test image X<sup>*</sup> to every image in training set X
- Find the one that matches it that is closest to it

#### Calculate distance

Euclidean distance:
$$ \sqrt{(x_2-x_1)^2+(y_2-y_1)^2} $$

Manhattan distance:
$$ |x_2-x_1|+|y_2-y_1| $$

### Assign label

- Assign the <b>label</b> of the (k=1) "nearest neighbour" to test data X<sup>*</sup> 

## Decision tree

- Capable of handling both classification and regression tasks
- Capable of dealing with complex, non-linear datasets
- Capable of working on raw feature data without the need for data pre-processing
- 'White Box' models : easy to interpret

HOWEVER, 
- Decision trees are prone to <b>overfitting</b>
	- To solve this, we can restrict the depth of a tree
	- Allow the tree to grow fully and then prune it to reach a compromise solution

### Root node

- Asks a binary question of the most distinguishing feature between two classes
- Feature that contains the most information

### Second or subsequent nodes/information feature

- Can use a condition on the feature with values above or below a certain threshold designated as one class and equal or  below to the other class

### Leaf nodes

- Bottom-most nodes where we make our final class designations

![[decision_boundary_decision_tree.png]]

- First level decision with a decision boundary that splits the dataset into 2

![[decision_boundary_2_decision_tree.png]]

- Second level decision on ear length that further splits the dataset, reducing impurities

## Classification and regression tree (CART)

- Binary tree
- Used for both classification and regression

![[recursive_decision_boundary_CART.png]]

- First, find the best way to split the dataset into 2 where each side has minimal impurity
- Each split is done with a decision boundary, and that decision boundary can be visualised in the binary tree
- Sections that are pure are considered leaf-nodes and regression will be stopped for those nodes
- Once the right/left half is done, the algorithm returns to the original root node and the other half will continue with the same process until we arrive at the final tree that perfectly divides the dataset

# Classifier evaluation

## Confusion Matrix

- A matrix where each cell contains a number that represents a population of labels that belong to a specific combination of predicted and actual classes
- Given the results of a classifier, we can formulate the confusion matrix by filling in the cells corresponding to predictions and actual ground truth labels

Video 2.501 example:

A binary classifier (Y --> {0,1}) is tested on N = 10 data samples
- correct 'ground truth' labels: 
	- Y<sup>GT</sup> = [1 0 0 1 0 0 1 1 1 1]
- classifier predicted output: 
	- Y<sup>P</sup> =   [1 1 1 1 1 0 1 1 1 1]

![[confusion_matrix.png]]

- Sum the diagonal elements and divide by the sum of all the matrix elements to get total accuracy
- Divide the diagonal element i with the sum of the elements in row i
- Per-class accuracy is also known as class recall

### Some terminologies

- Positive class: represents that the test case is classified to have the feature
- Negative class: represents that the test case is classified to not have the feature

- True Positives - Correctly identified with the feature
- True Negatives - Correctly identified without the feature
- False Positives - Falsely identified with the feature
- False Negatives - Falsely identified without the feature

- Recall - expresses how likely it is that a classifier will predict the correct value
$$ Recall = \frac{TP}{TP + FN} $$

- Precision - given a positive prediction, precision estimates how likely it is the prediction will be correct
$$ Precision = \frac{TP}{TP + FP} $$