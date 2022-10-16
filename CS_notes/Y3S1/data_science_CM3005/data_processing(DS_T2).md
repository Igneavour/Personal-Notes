---
tags: [data-type, numpy, numpy-memory-allocation, numpy-initialise-array, numpy-array-arithmetic, data-pre-processing, dataset-issue]
aliases: [DS T2, Data Science Topic 2, Data Processing]
---

# Reading resources for this topic

1. [NumPy: data types](https://numpy.org/doc/stable/user/basics.types.html)
2. [NumPy: array creation](https://numpy.org/doc/stable/user/basics.creation.html)

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

