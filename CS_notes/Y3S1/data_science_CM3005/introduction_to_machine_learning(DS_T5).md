---
tags: [supervised-learning, unsupervised-learning, classification, regression, clustering, dimensionality-reduction, principal-component-analysis, model-validation, holdout-set, validation-curve, learning-curve, grid-search, feature-engineering, missing-data, feature-pipeline]
aliases: [Data Science Topic 5, DS T5]
---

# Reading resources for this topic

1. [Chapter 11 Machine Learning](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=5750897)
2. [Chapter 5 pp 331-354](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=4746657)
3. [Generalization in machine learning pp 111-112](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=1323973)
4. [From holdout evaluation to cross-validation - Chapter 5 pp 126-129](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=1323973)
5. [Hyperparameters and model validation - pp 359-363](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=4746657)
6. [Selecting the best model - pg 363 - 375](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=4746657)
7. [Feature engineering - pg 375 - 382](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=4746657)

# Machine Learning approaches

## Supervised learning
From data to labels
- Classification
- Regression

## Unsupervised learning
Finding patterns in data
- Clustering
- Dimensionality reduction

# Classification

![[classification_DS.png]]

Goal: predict discrete labels

Find a line that separates the classes
- Learn the parameters of the line

Generalise to:
- New, unlabelled data
- Higher dimensions

# Regression

![[regression_DS.png]]

Goal: predict continuous labels

Model the relationship between variables
- Learn the parameters from examples

Generalise to:
- New, unlabelled data
- Higher dimensions, e.g: Feature1 = height, feature2 = weight , Label = age

# Clustering

![[clustering_DS.png]]

Goal: infer labels on unlabelled data

Use intrinsic structure to find groups
- Determine which points are related

Generalise to higher dimensions
- k-means algorithm

# Dimensionality reduction

![[dimensionality_reduction_DS.png]]

Goal: infer structure of unlabelled data

Transform data from high-dimensional space to low-dimensional space
- Preserve meaningful properties of the data

Useful for complex data sets
- Thousands of features
- Visualize using 2 or 3 dimensions

# Data representation in scikit-learn

Data as tables: two-dimensional grid
- Rows are instances (samples)
- Columns are attributes (features)

Example: Iris dataset

Feature matrix: X
- 2D: n_samples * n_features

Target array: y
- 1D: n_samples

Predict species based on measurements

![[splitting_dataset_DS.png]]

- splitting up dataset into features matrix and the target array or target vector

## Scikit-learn estimator API

Design principles
- Consistency
- Composition
- Sensible defaults

Overall process
- Choose a class of model e.g: random forest, linear regressor
- Set model hyperparameters - instantiate and configure your model
- Configure your data (X and y) - arrange into feature matrix and target vector
- Fit the model to your data - calculate the slope and their intercept
- Apply model to new unseen data - create grid of x values, visualize the results, plot raw data + model

# Linear regression 

- fit a line to some 2D data

- create some artificial data
	- y = mx + c
- Slope = ~2
- Intercept = -1

``` python
# set up the data
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y)
```

![[set_up_data_DS.png]]

``` python
# create a model
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
```

``` python
X = x[:, np.newaxis]
X.shape
```

(50, 1)

``` python
model.fit(X, y)
```

LinearRegression()

``` python
model.coef_
```

array([1.9776566])

``` python
model.intercept_
```

-0.9033107255311164

``` python
# create some new unseen data
xfit = np.linspace(-1, 11)

# arrange into a feature matrix
Xfit = xfit[:, np.newaxis]

# predict target array
yfit = model.predict(Xfit)

# visualize the raw data and the model fit
plt.scatter(x, y)
plt.plot(xfit, yfit)
```

![[visualize_data_DS.png]]

# Iris classification

- classify samples based on measurements

Using simple classifier: Naive Bayes

Steps followed:

1. Configure your data (x and y)
2. Fit the model to your data
3. Predict on test set

Goal: classify samples based on measurements
- Using simple classifier: Naive Bayes
- Fast, minimal configuration

``` python
# download the data
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()
```

![[iris_classification_DS.png]]

``` python
# extract the features matrix
X_iris = iris.drop('species', axis=1)
X_iris.shape
```

(150, 4)

``` python
# extract the target array
y_iris = iris['species']
y_iris.shape
```

(150,)

``` python
# split the data into a training set and test set
from sklearn.mode_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)

# create and train a model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
preds = model.predict(Xtest)

# evaluate the model
from sklearn.metrics import accuracy_score
accuracy_score(ytest, preds)
```

0.97384210523158

# Iris dimensionality reduction

- Try to reduce the number of dimensions, but retain essential features

Use principal components analysis:
4D --> 2D

Steps followed:
1. Choose a class of model
2. Instantiate the model with hyperparameters
<strike>3. Configure your data (x and y)</strike>
4. Fit the model to your data
5. Transform the data to 2 dimensions

``` python
# download the data
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()
```

![[iris_classification_DS.png]]

``` python
# extract the features matrix
X_iris = iris.drop('species', axis=1)
X_iris.shape
```

(150, 4)

``` python
from sklearn.decomposition import PCA 

# Choose the model class
model = PCA(n_components=2)

# Instantiate the model with hyoerparameters
model.fit(X_iris)

# Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)
```

PCA(n_components=2)

``` python
# Transform the data to two dimensions
X_2D = model.transform(X_iris)
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)
```

![[iris_PCA_DS.png]]

# Iris clustering

- we are trying to find structure in data e.g : meaningful groups

Using various techniques:
- kMeans
- Gaussian Mixture Models

Steps followed:
1. Choose a class of model
2. Instantiate the model with hyperparameters
<strike>3. Configure your data (x and y)</strike>
4. Fit the model to your data
5. Determine the clusters (and labels)

``` python
# download the data
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()
```

![[iris_classification_DS.png]]

``` python
# extract the features matrix
X_iris = iris.drop('species', axis=1)
X_iris.shape
```

(150, 4)

``` python
from sklearn.decomposition import PCA 

# Choose the model class
model = PCA(n_components=2)

# Instantiate the model with hyoerparameters
model.fit(X_iris)

# Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)

# Transform the data to two dimensions
X_2D = model.transform(X_iris)
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]

iris
```

![[iris_clustering_transformed_DS.png]]

``` python
# choose the model class
from sklearn.mixture import GaussianMixture

# Instantiate the model with hyperparameters
model = GaussianMixture(n_components=3, covariance_type='full')

``` python
# Fit to data. Notice y is not specified!
model.fit(X_iris)
```

GaussianMixture(n_components=3)

``` python
# Determine cluster labels
y_gmm = model.predict(X_iris)
iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species', col='cluster', fit_reg=False)
```

![[iris_clustering_seaborn_plot_DS.png]]

# Model validation

## Hold out sets

``` python
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target

# split the data into two halves
from sklearn.model_selection import train_test_split
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)

from sklearn.neighbours import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)

# fit the model to the first half
model.fit(X1, y1)
```

KNeighborsClassifier(n_neighbors=1)

``` python
# test it on the second half
y2_preds = model.predict(X2)

from sklearn.metrics import accuracy_score
```

output:

0.9066666666666666666 (instead of the previous 1.0)

### Using a holdout set

- hold back some of the data for testing
- so that the model has not 'seen' it
- better estimate of model's performance

HOWEVER
- we are 'wasting' part of our data (50%)
- a better solution: <b>cross-validation</b>

## Cross-validation

- perform multiple splits
- rotate the training and test portions
- calculate the mean performance

``` python
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target

# split the data into two halves
from sklearn.model_selection import train_test_split
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)

from sklearn.neighbours import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)

# fit the model to each half & predict
y1_preds = model.fit(X2, y2).predict(X1)
y2_preds = model.fit(X1, y1).predict(X2)

from sklearn.metrics import accuracy_score
accuracy_score(y1, y1_preds), accuracy_score(y2, y2_preds)
```

(0.96, 0.90666666666666666)

``` python
# cross_val_score allows us to set the number of iterations that we do, or otherwise called folds. the cv (cross-validation) value is the folds we want to perform. cross_val_score performs these iterations automatically for us compared to our previous manual iterations y1_preds and y2_preds
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv=5)
```

array([0.966666667, 0.96666667, 0.93333333, 0.93333333, 1.])

``` python
from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, X, y, cv=LeaveOneOut())
scores
```

![[leave_one_out_results_DS.png]]

# How to improve our models?

- A more complex model
- A less complex model
- More training data
- More/different features

## Considering bias and variance

![[bias_variance_tradeofff_DS.png]]

- The first one underfits the data --> high bias
	- training score R^2 = 0.70
	- validation score R^2 = 0.74
	- With high bias models, you tend to get similar performance on training vs test
- The second one overfits the data --> high variance
	- training score R^2 = 0.98
	- validation score R^2 = -1.8e^+09
	- With high variance models, you tend to get very different performance

![[function_of_model_complexity_DS.png]]

- performance is invariably better on training data
- high bias models are poor predictors for both training and test
- high variance models overfit to the training data
- the optimum may be somewhere in between

## Validation Curves

$$ \text{We will use a polynomial regression model with tunable parameter: degree} $$
$$ \text{First degree polynomial: } y = ax + b $$
$$ \text{Third degree polynomial: } y = ax^3 + bx^2 + cx + d $$
``` python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# return a pipeline with a polynomial preprocessor and simple linear regression
def PolynomialRegression(degree=2, **kwargs):
	return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

# create some data
import numpy as np

def make_data(N, err=1.0, rseed=1):
	# randomly sample the data
	rng = np.random.RandomState(rseed)
	X = rng.rand(N, 1) ** 2
	y = 10 - 1. / (X.ravel() + 0.1)
	if err > 0:
		y += err * rng.randn(N)
	return X, y

X, y = make_data(40)
```

``` python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # plot formatting

X_test = np.linspace(-0.1, 1.1, 500)[:, None]

plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()

# plot polynomials of degree 1, 3, 5
for degree in [1, 3, 5]:
	y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
	plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best')
```

![[plotting_curve_on_feature_matrix_DS.png]]

``` python
from sklearn.model_selection import validation_curve
degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), X, y, 'polynomialfeatures_degree', degree, cv=7)

plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
```

![[training_validation_curves_DS.png]]

- from the above image, we can deduce that the best performing degree of coefficient is around 3

``` python
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)
```

![[best_fit_curve_3_DS.png]]

## Size of data

- optimal model complexity will depend on the size of your training data:

``` python
X2, y2 = make_data(200)
plt.scatter(X2.ravel(), y2)
```

![[scatterplot_datasize200_DS.png]]

``` python
# We will compare a curve with datasize 40 (dotted background) and datasize 200 (solid
# line)
degree = np.arange(21)
train_score2, val_score2 = validation_curve(PolynomialRegression(), X2, y2, 'polynomialfeatures_degree', degree, cv=7)

# new curves with data size 200
plt.plot(degree, np.median(train_score2, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score2, 1), color='red', label='validation score')

# previous curves with data size 40
plt.plot(degree, np.median(train_score, 1), color='blue', alpha=0.3, linestyle='dashed')
plt.plot(degree, np.median(val_score, 1), color='red', alpha=0.3, linestyle='dashed')

plt.legend(loc='lower center')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
```

``` python

```

![[size_of_data_to_model_complexity_DS.png]]

Validation curves depend on both:
- model complexity
- size of training data

How should model behaviour change as a function of training data size?
- Training scores should always be higher than validation scores
- A model of given complexity will overfit a relatively small data set
- A model of given complexity will underfit a relatively large data set

![[learning_curves_schematic_DS.png]]

- as training set size increases, training score and validation score will get closer to each other
- however, there will be a point where increasing set size will not improve model performance. Hence, we will need to either changing the model or increasing complexity

### Learning curves

``` python
from sklearn.model_selection import learning_curve

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for i, degree in enumerate([2, 9]):
	N, train_lc, val_lc, = learning_curve(PolynomialRegression(degree),
										  X, y, cv=7,
										  train_sizes=np.linspace(0.3, 1, 25))

ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1],
			 color='gray', linestyle='dashed')

ax[i].set_ylim(0, 1)
ax[i].set_xlim(N[0], N[-1])
ax[i].set_xlabel('training size')
ax[i].set_ylabel('score')
ax[i].set_title('degree = {0}'.format(degree), size=14)
ax[i].legend(loc='best')
```

![[learning_curves_plot_DS.png]]

## Grid search

- Is there a way to automate the process of model optimization?
- use scikit-learn grid search

We can ask it to explore a 3D grid:
- Polynomial degree (int)
- Fit to the intercept (True/False)
- Whether to normalize (True/False)

``` python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# return a pipeline with a polynomial preprocessor and simple linear regression
def PolynomialRegression(degree=2, **kwargs)
	return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

# create some data
import numpy as np

def make_data(N, err=1.0, rseed=1)
	# randomly sample the data
	rng = np.random.RandomState(rseed)
	X = rng.rand(N, 1) ** 2
	y = 10 - 1. / (X.ravel() + 0.1)
	if err > 0:
		y += err * rng.randn(N)
	return X, y

# make a small data set
X, y = make_data(40)
```

``` python
from sklearn.model_selection import GridSearchCV

param_grid = {'polynomialfeatures_degree': np.arange(21),
			  'linearregression_fit_intercept': [True, False],
			  'linearregression_normalize': [True, False]}

grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)

grid.fit(X, y)
grid.best_params_
```

{'polynomialfeatures_degree': 4,
 'linearregression_fit_intercept': False,
 'linearregression_normalize': True}

# Feature engineering

Up to now we have assumed that our data is in a readily usable format. However, real-world data is rarely like this!
- How can we transform information into usable data?

Dealing with non-numerical data, e.g
- Categorical data
- Textual data
- Image data

- Creating derived features
- Dealing with noisy data, e.g missing values

## Example of feature engineering code (categorical data)

``` python
data = [
		{'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
		{'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
		{'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
		{'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]
```

``` python
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)
```

array([[0, 1, 0, 850000, 4],
		 [1, 0, 0, 700000, 3],
		 [0, 0, 1, 650000, 3],
		 1, 0, 0, 600000, 2]]])

``` python
vec.get_feature_names()
```

['neighborhood=Fremont',
'neighborhood=Queen Anne',
'neighborhood=Wallingford',
'price',
'rooms']

``` python
# previous example where sparse=False will make the output unreadable. Therefore,
# setting sparse=True will make it more readable as seen below
vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data)
```

<4x5 sparse matrix of type '<class 'numpy.int32'>' 
	with 12 stored elements in Compressed Sparse Row format>

## Example of feature engineering code (textual data)

``` python
sample = ['the stars are not wanted now; put out every one',
		  'pack up the moon and dismantle the sun',
		  'pour away the ocean and sweep up the wood',
		  'for nothing now can ever come to any good']
```

``` python
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(sample)
X
```

<4x29 sparse matrix of type '<class 'numpy.int64'>'
	with 34 stored elements in Compressed Sparse Row format>

``` python
import pandas as pd
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
```

![[pd_textual_data_DS.png]]

``` python
from sklearn.feature_extraction.text import TfidVectorizer
vec = TfidVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
```

![[tfid_vectorizer_text_DS.png]]

## How to choose the right feature?

Sometimes the choice is not so simple

Example: building a spam classifier
- Does the email contain the word 'Viagra' or 'Lottery'
- Frequency of certain characters, e.g. ! or $
- Email address of the sender

How do we choose?
- Experience
- Domain expertise
- Experimentation

Sometimes features can interact in unexpected ways, e.g : house prices
- Location
- Size

We may need to consider combined features
Or mathematically derived features

### Feature selection example

``` python
# create some non-linear data points
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y)
```

![[feature_selection_plot_DS.png]]

``` python
# attempt to fit a straight line
from sklearn.linear_model import LinearRegression
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit)
```

![[attempt_str_line_on_non_linear_DS.png]]

``` python
# add an extra feature representing x^2
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X2 = poly.fit_transform(X)
print(X2)
```

[[1.   1.]
[2.   4.]
[3.   9.]
[4.   16.]
[5.   25.]]

``` python
# Now the predictions are much closer
model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit)
```

![[adding_extra_features_DS.png]]

## Handling missing data

Real-world data is rarely clean or homogeneous

Missing data conventions
- Use a mask, e.g. a Boolean flag
- Use a sentinel value, e.g. -9999

Panda uses:
- None (a Python object)
- NaN (a float64)

### Dealing with null values

Detection:
- <code>isnull()</code>
- <code>notnull()</code>
Returns a Boolean mask over the data

Drop null values:
- <code>dropna()</code>
Removes NA values (how + thresh params control # of nulls to allow)

Fill null values:
- <code>fillna()</code>
Replaces NA values (e.g. with single value, previous value, etc)

### Imputation of missing values code

``` python
# create a 2D array with missing values
import numpy as np
from numpy import nan

X = np.array([[ nan, 0, 3],
			 [ 3, 7, 9],
			 [ 3, 5, 2],
			 [ 4, nan, 6],
			 [ 8, 8, 1]])
y = np.array([14, 16, -1, 8, -5])
```

``` python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='most_frequent')
X2 = impt.fit_transform(X)
X2
```

array([[ 3, 0, 3],
		 [ 3, 7, 9],
		 [ 3, 5, 2],
		 [ 4, 3, 6],
		 [ 8, 8, 1]])

``` python
# fit a linear regression model to the data
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
model.predict()
```

array([ 8.21123945, 14.1271657, 3.06186139, 12.49966681, -5.89993336])

``` python
model.score(X2, y)
```

0.7788468075082486

## Feature pipelines

- basically combines all steps taken in the data science project into a object to streamline the process and reduce the chances of error

### Common processing pipeline

- Impute missing values (e.g using median)
- Transform features (e.g using a polynomial)
- Fit a linear regression

``` python
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
model = make_pipeline(SimpleImputer(strategy='mean'),
					  PolynomialFeatures(degree=3),
					  LinearRegression())
```

``` python
model.fit(X, y)
preds = model.predict(Xtest)
print(preds)
```

[15.05233793   16.72892509  -5.3003913  16.16457818  -11.11390686]

``` python
model.score(Xtest, ytest)
```

0.6141482204733837