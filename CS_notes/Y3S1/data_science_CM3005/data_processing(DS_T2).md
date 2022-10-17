---
tags: [data-type, numpy, numpy-memory-allocation, numpy-initialise-array, numpy-array-arithmetic, data-pre-processing, dataset-issue, advanced-indexing, structured-data, numpy-statistics, measures-of-central-tendency, measures-of-spread, linear-algebra, scalar, vector, scalar-multiplication, vector-addition, dot-product, matrix, determinant, rank, trace, inverse-matrix, linear-equation, series, dataframe, pandas, time-series]
aliases: [DS T2, Data Science Topic 2, Data Processing]
---

# Reading resources for this topic

1. [NumPy: data types](https://numpy.org/doc/stable/user/basics.types.html)
2. [NumPy: array creation](https://numpy.org/doc/stable/user/basics.creation.html)
3. [NumPy, Array Objects: Indexing](https://numpy.org/doc/stable/reference/arrays.indexing.html)
4. [NumPy, Routines: Statistics](https://numpy.org/doc/stable/reference/routines.statistics.html)
5. [Linear Algebra Chapter 4: Matrices](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=1920019)
6. [Linear Algebra Chapter 6: Determinants](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=1920019)
7. [Python data science handbook Chapter 2: Introduction to NumPy](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=4746657)
8. Read the following sections:
	- Introduction
	
	- Overview
	
	- Timestamps vs. time spans
	
	- Converting to timestamps
	
	- Generating ranges of timestamps
	
	- Timestamp limitations
	
	- Indexing is here but cover this later
	
	- Time/date components
	
	- DateOffset objects
	
	- Time Series-Related Instance Methods
	
	- Resampling
	[Pandas User Guide: Time series/date functionality](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
9. [pandas getting started: intro to data structures](https://pandas.pydata.org/docs/user_guide/dsintro.html) 
10. [pandas API reference: Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html)
11. [pandas API reference: DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
12. [Python data science handbook chapter 3: Data Manipulation with Pandas](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=4746657&query=Python+data+science+handbook%3A+essential+tools+for+working+with+data.)

# Data types

- Boolean (logical)
- Numerical
- Textual
- Images
- Sound
- Video
- Digital signals
- Others

Since computers are digital/binary machines, storing and processing any type of data is actually done on their corresponding binary representations.

Types of binary representations:
- One's and two's complement
- IEEE standardised floating point number

# Introduction to NumPy

## Memory allocation

### 1D array of integers using the 'empty' method

``` Python
import numpy as np

# Number of elements
NumberofElements = 50
# Allocate memory
MyArray = np.empty(shape=NumberofElements, dtype=int)
```

- This method does not initialise the elements of the array
- The elements  have random values found on the memory locations at the time of the allocation

### 2D array of integers using the 'empty' method

``` python
import numpy as np

# Define dimensions
NumberOfRows = int(10)
NumberOfColumns = int(50)
# Allocate memory
MyArray = np.empty(shape=(NumberOfRows, NumberOfColumns), dtype=int)
```

- The result is an array of 10x50 un-initialised elements.

## Initialise arrays

- 1D array of integers using a random number generator
- We can specify the number of elements, their range and type:

``` python
import numpy as np

# Initialise the random number generator
np.random.seed(0)
# Initialise array with 10 elements with random values between -5 and 5
MyArray = np.random.randint(-5,5,10, dtype=int)
```

### Initialise all elements with a specific value and print the array

- Zero

``` python
# Number of elements
NumberOfElements = 10
# Allocate memory and initialise the elements with zeros
MyArray = np.zeros(shape=NumberOfElements, dtype=int)
# Print the array
print(MyArray)
```
Output: [0 0 0 0 0 0 0 0 0 0]

- Ones

``` python
# Number of elements
NumberOfElements = 10
# Allocate memory and initialise the elements with ones
MyArray = np.ones(shape=NumberOfElements, dtype=int)
# Print the array
print(MyArray)
```
Output: [1 1 1 1 1 1 1 1 1 1]

- Arbitrary value

``` python
# Number of elements
NumberOfElements = 10
# Allocate memory and initialise the elements with 7
MyArray = np.ones(shape=NumberOfElements, fill_value=7, dtype=int)
# Print the array
print(MyArray)
```
Output: [7 7 7 7 7 7 7 7 7 7]

- Random integers

``` python
# Number of elements
NumberOfElements = 10
# Allocate memory and initialise the elements with random values
MyArray = np.random.randint(0, 100, NumberOfElements)
# Print the array
print(MyArray)
```
Output: [82 79 45 57 3 43 98 23 14 15]

## Array arithmetic

### Sum and Average

``` python
# Allocate memory the array and initialise the elements with random values
MyArray = np.random.randint(0, 100, NumberOfElements)
# Print the array
print(MyArray)
# Display the sum and the average
print('Sum:', MyArray.sum())
print('Average:', np.average(MyArray))
```
Output: 
[86 90 24 13 24 93 85 51 80 13]
Sum: 559
Average: 55.9

### Finding sum per column and per row using parameter 'axis' to specify direction of the summation

``` python
# Summation by columns
SumByColumn = np.sum(MyArray, axis=0)
print('By column', SumByColumn)
# Summation by rows
SumByRow = np.sum(MyArray, axis=1)
print('By row:', SumByRow)
```
Output:
![[sum_axis_array_arithmetic.png]]

- Per-column and per-row operations not limited to 'sum'
- Available for min, max and average

## Data pre-processing

- Acquired data usually lack the structure, consistency and format necessary for computer algorithms to perform their tasks successfully

### Common issues with data:

- missing values
- type inconsistency or type mismatch
- duplicating values, including entire rows and columns

Pre-processing the data will resolve most of the issues mentioned above, bringing a database to <b>first normal form (1NF)</b>.

### Pre-processing process

- A common pre-processing task is to extract useful data from an existing datset
- This amounts to generating a sub-array based on an original array
- This can be done wiht NumPy array by using the <b>advanced indexing</b>

### Advanced indexing

- Allows us to build new arrays based on existing arrays
- We can do this by specifying a set of elements which will be included in the new array
- This can be applied to arrays with different dimensions

- Extract a sub-array by specifying start and end index of the elements
``` python
# Alternate way to initialise array but NOT practical
A = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Extract sub-array B
B = A[2:5]
```
Output: [2 3 4]

- The resulting array does not need to be comprised of all elements between the start and end index. We can set a step (stride)
``` python
# Initialise array A with 10 values 0 to 9
A = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Extract sub-array B
B = A[2:8:2]
# Display the new array
print(B)
```
Output: [2 4 6]

- Example below does not specify start and end for B and only start for C
``` python
A = np.array[(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]
B = A[::2]
C = A[1::2]
```
Output:
[0 2 4 6 8]
[1 3 5 7 9]

- 2D array advanced indexing
``` python
# Initialise 2D array of 3x3 elements
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
# Extract and display sub-array
B = A[2:3, 1:3]
C = A[1:3, 2:3]
# Display the array
print('A:',A)
print('B:',B)
print('C:',C)
```
Output:
![[2D_advanced_indexing.png]]

## Advanced indexing: lecture task

![[advanced_indexing_task.png]]

Here is my solution and output:

``` python
import numpy as np
A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print('A:',A)
B = A[1:4,:3]
print('B:',B)
C = A[::2,:]
print('C:',C)
```
Output:
``` python
A: [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]
B: [[ 5  6  7]
 [ 9 10 11]
 [13 14 15]]
C: [[ 1  2  3  4]
 [ 9 10 11 12]]
 ```

- In most cases, data science deals with data organised in the form of a table, similarly to XML and spreadsheets.
- It is not uncommon for these tables to contain rows and columns which prevent further processing.
- This may be due to various reasons, including missing values, type mismatch and others.
- Removing, or 'dropping', problematic rows and columns is the first and, in some cases, the only step of the pre-processing task.
- This can be achieved with other libraries (pandas), which support alternative approaches with similar results.

## Structured data

Datasets can be split into two groups, based on the type of their data points:
1. Datasets containing data points of the same type
> This is usually numerical, either integer or real

2. Datasets containing data points of multiple data types
> For example, numerical, textual, Boolean

The first step is to define the array and allocate the necessary memory:  
<code>A = np.empty(shape=10, dtype=[('Title', 'U10'), ('Year', int), ('Price', float)])</code>

- Structured data is usually stored as CSV, XML or a spreadsheet table
- Other libraries, such as pandas, handle this task differently
- Once the array has been defined as seen above, it can be initialised with specific data:

``` python
# Initialise the array
A[0] = ('Mathematics', 2012, 12.5)
A[1] = ('Science', 2015, 25.75)
A[2] = ('History', 2010, 15.5)
print(A)
```
Output:
``` python
[('Mathematics', 2012, 12.5)
('Science', 2015, 25.75)
('History', 2010, 15.5)]
```

## Statistics with NumPy

Statistics uses two main measurements to describe a dataset, namely:
- measures of central tendency
- measures of spread

### Measures of central tendency

> Measures of central tendency evaluate a central value, around which the data cluster around

There are 3 popular measures of central tendency, namely:
- Mean (or average/arithmetic average)
- Median (splits dataset into two halves)
- Mode (the data point which appears most frequently)

``` python
import numpy as np
from scipy import stats as st
Mean = np.mean()
Median = np.median()
# numpy does not have mode function
Mode = st.mode(S)
```

### Measures of spread

> Measures of spread evaluate how data are spread around certain central values

3 common measures of spread are:
- range - the difference between the maximum and the minimum value from the dataset
- standard deviation - measures the average distance of the data points from the mean value
- variance

Since variance = sd<sup>2</sup>, we only need to calculate the standard deviation.

- In many cases a dataset consists of multiple series of data organised into tables
- These series might be organised into rows or columns
- This brings the need to extract statistical information from the dataset (and the table) per-row or per-column, rather than overall

``` python
# Dimensions or the dataset
NumberOfRows = int(5)
NumberOfColumns = int(6)

# Allocate memory
MyArray = np.empty(shape=(NumberOfRows, NumberOfColumns), dtype=int)

# initialise a dataset with random values
# Display the dataset as a table
for i in range(0, NumberOfRows):
	for j in range(0, NumberOfColumns):
		MyArray[i,j] = np.random.randint(0, 10)
		print(MyArray[i,j],'',end='')
		print('')
```

- Considering each column as an independent data series, we can extract statistical information per column:

``` python
MeanByColumn = np.mean(MyArray, axis=0)
MedianByColumn = np.median(MyArray, axis=0)
ModeByColumn = st.mode(MyArray, axis=0)
SD = np.std(MyArray, axis=0)
```

Output:
``` python
3 3 7 8 9 2
7 3 5 4 0 6
2 4 8 5 3 5
5 3 0 2 9 1
6 9 1 9 1 7

Mean by column: [4.6 4.4 4.2 5.6 4.4 4.2]
Median by column: [5. 3. 5. 5. 3. 5.]
Mode by column: ModeResult(mode=array([[2,3,0,2,9,1]]), count=array([[1,3,1,1,2,1]]))
Standard deviation by column: [1.8547237 2.33238076 3.18747549 2.57681975 3.87814389 2.31516738]
```

# Introduction to linear algebra

Some of the key concepts in linear algebra:
- linear combinations
- systems of linear equations
- linear transformations (aka linear maps)

## Scalars

- Scalars are values which only have magnitude
- Examples are real numbers or any of their subsets like natural, integer and rational
- Complex numbers are usually treated as scalars although in some context they can be considered as 2D vectors

## Vectors

![[2D_vectors_diagram.png]]

- In 2D and 3D spaces, a vector is a quantity represented by an arrow with both direction and magnitude
- In linear algebra, the concept of a vector is generalised for any dimension (n)
> v = (a<sub>1</sub>, a<sub>2</sub>, ... a<sub>n</sub>)
- The algebraic notion of a vector corresponds to the array data type, as supported by a number of programming languages

### Vectors main operations

1. Scalar multiplication:
	- Scalar: a
	- Vector: v = (v<sub>1</sub>, v<sub>2</sub>, ... , av<sub>n</sub>,)
	- av = a(v<sub>1</sub>, v<sub>2</sub>, ... , av<sub>n</sub>,) = (av<sub>1</sub>, av<sub>2</sub>, ... , av<sub>n</sub>,)
2. Vector addition:
	- Vectors: x, y, z (same dimension)
	- x = (x<sub>1</sub>, x<sub>2</sub>, ... , x<sub>n</sub>,), y = (y<sub>1</sub>, y<sub>2</sub>, ... , y<sub>n</sub>,)
	- z = x + y = (x<sub>1</sub> + y<sub>1</sub>, x<sub>2</sub> + y<sub>2</sub>, x<sub>n</sub> + y<sub>n</sub>)
3. Dot product:
	- Vectors: x, y, z (same dimension)
	- x = (x<sub>1</sub>, x<sub>2</sub>, ... , x<sub>n</sub>,), y = (y<sub>1</sub>, y<sub>2</sub>, ... , y<sub>n</sub>,)
	- z = x · y = x<sub>1</sub>y<sub>1</sub> + x<sub>2</sub>y<sub>2</sub> + x<sub>n</sub>y<sub>n</sub>
	- The result of the dot product is a single value, a scalar

#### Scalar multiplication 

``` python
# Define a scalar C
C = int(5)
# Define three-dimensional vector A
A = np.array([1,2,3], dtype=int)
# Multiply the scalar and the vector
B = C*A
# Display the result on the screen
print(B)
```

Output:
[5, 10, 15]

#### Vector addition

``` python
# Define 3D vectors X and Y
X = np.array([3,4,5], dtype=int)
Y = np.array([5,6,7], dtype=int)
# Multiply the vectors
Z = np.add(X,Y)
# Display vector Z on the screen
print(Z)
```

Output:
[8, 10, 12]

#### Dot product

``` python
# Define 3D vectors X and Y
X = np.array([1,2,3], dtype=int)
Y = np.array([5.10.15], dtype=int)
# Calculate the dot product D
D = np.dot(X, Y)
# Display dot product D on the screen
print('D:', D)
```

Output:
D: 70

## Matrices

- A matrix is a rectangular table of components (scalars or numbers)
- The scalars are arranged in rows and columns by using a two-index notation
- The following is a generalised matrix M with m x n components:

![[generalised_matrix_M.png]]

- Similar to vectors, matrices can be defined as 2D NumPy arrays

### Matrices and NumPy: determinant

- Every square matrix has a <b>determinant</b>
- The determinant is a single value, which has important mathematical meanings
- One of the key questions is whether the determinant has zero or non-zero value
> A matrix with a determinant of zero does not have an inverse. If this matrix represent coefficients of a system of linear equations, this system either does not have a solution or has infinite number of solutions.
- Calculating a determinant requires multiple additions and multiplications, and in practice it is usually done by using a software tool

- The following code initialises a 3 by 3 matrix (array) and calculates its determinant

``` python
# Initialise a matrix M with 3 by 3 elements
M = np.array([[1,2,3],[2,3,4],[2,5,6]])
# Calculate the determinant of M
D = np.linalg.det(M)
# Display the matrix
print(M)
# Display the determinant
print('D:', D)
```

Output:
``` python
[[1 2 3]
 [2 3 4]
 [2 5 6]]

D: 2.0
```

- If one of the row (or column) is a <b>linear combination</b> of another row (or column), the value of the determinant is 0:

``` python
M = np.array([[1,2,3],[2,4,6],[2,5,6]])
# Calculate the determinant of M
D = np.linalg.det(M)
# Display the matrix
print(M)
# Display the determinant
print('D:', D)
```

Output:
``` python
[[1 2 3]
 [2 3 6]
 [2 5 6]]

D: 0.0
```

### Matrices and NumPy: rank

- A square matrix with non-zero determinant has rank equals to its dimensions
- A matrix does not need to be square to have a rank, we can form a determinant from a known square matrix by removing rows and columns

``` python
M = np.array([[1,2,3],[2,3,4],[2,5,6]])
# Calculate the rank of M
R = np.linalg.matrix_rank(M)
# Display the matrix
print(M)
# Display the rank
print('R:', R)
```

Output:
``` python
[[1 2 3]
 [2 3 4]
 [2 5 6]]

R: 3
```

- We can lower the rank of a matrix by defining a row as a linear combination of another row, as illustrated in the example below:

``` python
M = np.array([[1,2,3],[2,3,4],[4,6,8]])
# Calculate the rank of M
R = np.linalg.matrix_rank(M)
# Display the matrix
print(M)
# Display the rank
print('R:', R)
```

Output:
``` python
[[1 2 3]
 [2 3 4]
 [4 6 8]]

R: 2
```

Since the 3rd row is a linear combination of the 2nd row, the corresponding determinant is zero. If we form a 2D determinant from the 1st two rows and the 1st two columns, its value will not be zero. Thus, the rank of the matrix above is 2 which is the dimension of the largest non-zero determinant.

### Matrices and NumPy: trace

- The trace of a matrix is the sum of all elements on the main diagonal. The following example illustrates the calculation of trace:

``` python
M = np.array([[1,2,3],[2,3,4],[4,6,8]])
# Calculate the trace of M
T = np.trace(M)
# Display the matrix
print(M)
# Display the trace
print('T:', T)
```

Output:
``` python
[[1 2 3]
 [2 3 4]
 [4 6 8]]

T: 12
```

### Matrices and NumPy: matrix multiplication

- The result of the multiplication is also a matrix
- The multiplication follows specific rules:
	- It is based on the dot product
	- It requires the number of columns from the first matrix (j) to be equal to the number of rows in the second matrix (k):
	> $$ C_{il} = A_{ij} * B_{kl} ; j = k $$
	
	![[matrix_multiplication.png]]

c<sub>11</sub> = a<sub>11</sub> * b <sub>11</sub> + a<sub>12</sub> * b <sub>21</sub> + ... + a<sub>1n</sub> * b <sub>n1</sub>

### Matrices and NumPy: inverse matrix

- A square matrix with non-zero determinant has an inverse
- In the case of real numbers, where the multiplicative inverse of r is 1/r (for example, the multiplicative inverse of 2 is 1/2)
- The inverse of a matrix M is usually denoted with M<sup>-1</sup>
- M x M<sup>-1</sup> = M<sup>-1</sup> x M = I
- I is the identity matrix
- Multiplying any square matrix with I does not change the matrix

![[identity_matrix.png]]

``` python
M = np.array([[1,2,3],[2,3,4],[2,5,6]])
# Calculate the inverse of M
M1 = np.linalg.inv(M)
# Display the matrix
print('M:', M)
# Display the inverse matrix
print('M1:', M1)
```

Output:
``` python
M [[1 2 3]
   [2 3 4]
   [2 5 6]]
M1: [[-1. 1.5 -0.5]
	 [-2 0. 1.]
	 [2. -0.5 -0.5]]
```

## Linear equations

![[linear_equations.png]]

x + 2y + 3z = 1
2x + 3y + 4z = 4
2x + 5y + 6z = 7

- We can express the system of linear equations by using matrices:

$$
\left(\begin{array}{cc} 
1 & 2 & 3\\
2 & 3 & 4\\
2 & 5 & 6
\end{array}\right)
\left(\begin{array}{cc} 
x\\ 
y\\
z
\end{array}\right)
=
\left(\begin{array}{cc}
1\\
4\\
7
\end{array}\right)
$$

And by using the following matrix notation:

$$
MX = C, where: M =

\left(\begin{array}{cc} 
1 & 2 & 3\\
2 & 3 & 4\\
2 & 5 & 6
\end{array}\right)
, X =
\left(\begin{array}{cc} 
x\\ 
y\\
z
\end{array}\right)
, C =
\left(\begin{array}{cc}
1\\
4\\
7
\end{array}\right)
$$

By performing these steps:

MX = C
M<sup>-1</sup>MX = M<sup>-1</sup>C
IX = M<sup>-1</sup>C
X = M<sup>-1</sup>C

The code below performs the same operations:

``` python
M = np.array([[1,2,3],[2,3,4],[2,5,6]])
# Calculate the inverse of M
M1 = np.linalg.inv(M)
# Initialise matrix C
C = np.array([[1], [4], [7]])
# Calculate X = M1*C
X = np.matmul(M1, C)
# Display matrix X
print('X:', X)
```

Output:
``` python
X: [[1.5]
	[5.]
	[-3.5]]
```

$$
X =
\left(\begin{array}{cc} 
x\\ 
y\\
z
\end{array}\right)
=
\left(\begin{array}{cc}
1.5\\
5\\
-3.5
\end{array}\right)
, x = 1.5, y = 5, z = -3.5
$$

1.5 + 2 * 5 - 3 * 3.5 = 1
2 * 1.5 + 3 * 5 - 4 * 3.5 = 4
2 * 1.5 + 5 * 5 - 6 * 3.5 = 7

- This is not the only approach to solving the linear equations:
	- Gaussian elimination
	- Cramer's rule

# Series and dataframes in pandas

## Core data structures in pandas

Series
- one-dimensional labelled array
- data can be any type

DataFrame
- two-dimensional labelled table
- columns can be of different type

Refer to video in coursera for demonstrations in pandas usage:
<form action="https://www.coursera.org/learn/uol-cm3005-data-science/lecture/WekhK/2-201-series-and-data-frames-in-pandas">
	<input type="submit" value="Video 2.201 Series and data frames in pandas">
</form>

## Time-series data

Time-series is a sequence of variable measurements, indexed by time

### Uses and applications
- business and finance
	- analysis of stock price, costs, profit, units sold ...
- organisational
	- monitoring process or quality, workload projections
- government and policy making
	- changes in economic, health, crime, statistics
- science and engineering
	- signal processing, energy efficiency, cell mutation

### Looking for trends

- seasonal variation
- significant events
	- natural disasters, changes of government, tax regulation
- correlations
	- do variables change in similar ways over time?
- understanding the past
- predicting the future

### Different kinds of pattern

- variability
- rate of change
- co-variance and correlation
- cycles
- exceptions

### Sampling rate

- some variable are constantly changing over time
	- temperature, physiological response, stock market
- rate of measurement is important

Example:

Temperature
> one measurement per hour, or one measurement per second?

Stock market
> trend over a whole year, or changes over milliseconds

### Time related concepts in pandas

Date times: A specific date and time with timezone support
Time deltas: An absolute time duration
Time spans: A span of time defined by a point in time and its associated frequency
Date offsets A relative time duration that respects calendar arithmetic

![[panda_datatypes.png]]

Note: Need to log into coursera first before you can access the links

<form action="https://www.coursera.org/learn/uol-cm3005-data-science/lecture/PADBm/2-205-representing-time">
	<input type="submit" value="Video 2.205 Representing Time">
</form>

<form action="https://www.coursera.org/learn/uol-cm3005-data-science/lecture/GurPe/2-206-time-series-analysis">
	<input type="submit" value="Video 2.206 Time Series Analysis">
</form>