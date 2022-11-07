---
tags: [descriptive-statistic, data-critical, population, sample, inferential-statistic, univariate-analysis, multivariate-analysis, central-tendency, quartiles, box-plot, tukey-plot, histogram, bar-graph, standard-deviation, z-score-normalisation, quantile-plot, quantile-quantile-plot, skewness, kurtosis, leptokurtic-distribution, platykurtic-distribution, confidence-band]
aliases: [DS T4, Data Science Topic 4]
---

# Reading resources for this topic

1. [Analysing quantitative data pg 46-51](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=354948)
2. [Central tendency pg 22-24](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=1729064)
3. [distribution of the data pg 24-36](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=1729064)
4. [chapter 2 data visualisation, section 2.3 tables and section 2.4 univariate data visualisation pg 32-49](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=1729064)
5. [chapter 4 exploring data visually, pg 143-153 (visualising categorical data) and pg 189-199 (distributions)](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=1158630)
6. [4.2.1 scatterplots, 4.2.2 summary chart and table, 4.2.3 cross-classification tables, 4.3 calculating metrics and relationships](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=1729064)
7. [pg 49-59](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=427614)

# Descriptive statistics

- also called <b>summary statistics</b>
- provide quantitative descriptions or summaries of data
- helps us understand data
- helps us decide what data to present and how
- are the fundamental quantities visualised in many plot types

# Data critical

- vital that we know what/who any data represents
- how was it collected?
- why was it collected?
- who was involved?
- what valid conclusions can we draw?

# Sample and population

> Population is the collection of all individuals or items under consideration
> Sample is the part of population from which information is obtained

# Descriptive vs Inferential statistics

> descriptive statistics describe, show, or summarise data in a meaningful way such that patterns might emerge from the data

> inferential statistics techniques that allow us to:
> 1. use samples to make generalisation about population
> 2. use data about the past to predict the future

# Univariate and Multivariate analysis

> univariate analysis understand the shape, size and range of quantitative values (a single variable)
> here, we look at measures of central tendencies such as mean value, and measures of spread such as range

> multivariate analysis explore the possible relationships between different combinations of variables and variable types (many variables)

# Measure of central tendency

- mode
	- most frequent score in a data set
- median
	- middle score for a set of data arranged in order of magnitude
- mean ('average')
$$ \bar X = \frac{\sum_{i=1}^{i=n}X_i}{n} $$

# Appropriate statistics for different variable types

|                                          | Nominal | Ordinal | Interval | Ratio |
| ---------------------------------------- | ------- | ------- | -------- | ----- |
| mode, frequency distribution             | Yes     | Yes     | Yes      | Yes   |
| median and percentiles                   | No      | Yes     | Yes      | Yes   |
| mean, standard deviation, standard error | No      | No      | Yes      | Yes   |
| ratio, or coefficient of variation       | No      | No      | No       | Yes      |

# Mode

``` python
df = pd.read_csv('./data/transport-data.csv', index_col = 0, 
				dtype = {'transport' : 'category'})
df.head(6)
```

Output:

![[mode_df_output.png]]

## Frequency distribution

``` python
counts = df['transport'].value_counts()
counts
```

Output:

![[mode_freq_distribution.png]]

## Plotting the bar chart

``` python
# Plot the counts as a bar chart
ax = counts.plot.bar(rot=0)

ax.set_title('Counts of preferred means of transport')
ax.set_xlabel('vehicle type')
ax.set_ylabel('frequency')

# Format counts as integers
ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter(places=0))

# Find the mode so we can label it on the plot
mode = df['transport'].mode()[0]

# Find the index of the mode in the plot
mode_pos = counts.index.get_loc(mode)
ax.annotate('mode={}'.format(mode), xy=(mode_pos + 0.25, 14), 
		   xytext=(mode_pos + 0.7, 15), 
		   arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

Output:

![[mode_bar_graph.png]]

## Plotting the histogram

``` python
# Read the data file
df = pd.read_csv('./data/class-data.csv', index_col=0)

# Make one bin = 5cm
bin_size = 5
start = round(df['height'].min() / bin_size) * bin_size
end = df['height'].max() + bin_size
bins = np.arange(start, end, bin_size)
ax = df['height'].plot.hist(bins=bins)

ax.set_title('Histogram of the heights of people in the class')
ax.set_xlabel('height (cm)')
ax.set_ylabel('frequency')

mode = pd.cut(df['height'], bins).mode()[0]
ax.annotate('mode={:0.0f}-{:0.0f}'.format(mode.left, mode.right), 
		   xy=(mode.right, 7), 
		   xytext=(mode.right + 6, 7.3),
		   arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

Output:

![[mode_histogram.png]]

# Median

``` python
df = pd.DataFrame({'age': [
   65, 55, 89, 56, 35, 14, 56, 55, 87, 45
]})
df['age'].median()
```

Output:
55.5

## Plotting the histogram

``` python
# Read the data file
df = pd.read_csv('./data/class-data.csv', index_col=0)

# Make one bin = 5cm
bin_size = 5
start = round(df['height'].min() / bin_size) * bin_size
end = df['height'].max() + bin_size
bins = np.arange(start, end, bin_size)
ax = df['height'].plot.hist(bins=bins)

ax.set_title('Histogram of the heights of people in the class')
ax.set_xlabel('height (cm)')
ax.set_ylabel('frequency')

median = df['height'].median()
ax.axvline(median, color='black', linestyle='dashed', linewidth=1)
ax.annotate('median={:0.1f}'.format(median), xy=(median, 7), 
		   xytext=(median + 6, 7.3), 
		   arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

Output:

![[median_histogram.png]]

# Mean

``` python
# Read the data file
df = pd.read_csv('./data/class-data.csv', index_col=0)

# Make one bin = 5cm
bin_size = 5
start = round(df['height'].min() / bin_size) * bin_size
end = df['height'].max() + bin_size
bins = np.arange(start, end, bin_size)
ax = df['height'].plot.hist(bins=bins)

ax.set_title('Histogram of the heights of people in the class')
ax.set_xlabel('height (cm)')
ax.set_ylabel('frequency')

mean = df['height'].mean()
ax.axvline(median, color='black', linestyle='dashed', linewidth=1)
ax.annotate('mean={:0.1f}'.format(mean), xy=(mean, 7), 
		   xytext=(mean + 6, 7.3), 
		   arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

Output:

![[mean_histogram.png]]

# Why sometimes median is better

| staff  | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10  |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| salary | 15k | 16k | 18k | 18k | 21k | 23k | 30k | 35k | 90k | 95k |

- mean salary of these 10 staff is 36.1k
- median is 22k
- median is not skewed so much by very large or small values

# Quartiles

> Values which split datasets into 4 equal parts. 
> Dataset must be ordered before identifying the quartiles, mostly ascending order.> 

Quartiles are 3 values, usually denoted Q1, Q2, Q3: 
- Q1: median of the first half or 25th percentile
- Q2: median of the entire dataset or 50th percentile
- Q3: median of the second half or 75th percentile

## Computing Q1 and Q3

- include Q2 data point in each 'half' or not?
	- yes for odd number of data items
	- default in pandas and R

## Quartiles in python ( quantile() )

``` python
s = pd.Series([6, 7, 15, 36, 39, 40, 
			  41, 42, 43, 47, 49], name='data')
s.quantile([0.25, 0.5, 0.75])
```

Output: 

![[quartiles_python.png]]

## Interquartile range

> difference between the upper and lower quartiles
> IQR = Q3 - Q1

## Box Plot

``` python
# Read the data file
df = pd.read_csv('./data/class-data.csv', index_col = 0)

ax = df['height'].plot.box()
ax.set_title('Distribution of the heights of people in the class')

ax.set_ylabel('cm')
plt.show()
```

Output:

![[quartiles_box_plot.png]]

## Tukey plot

- whiskers represent most extreme non-outlier data
- dots represent outliers
- limit is 1.5 IQR
	- whiskers extend to the maximum value within 1.5 IQR from Q3, and to the minimum value within 1.5 IQR from Q1

![[tukey_plot.png]]

## Quantile plot

``` python
# Random data from Myatt and Johnson (2009) p 46
x = pd.Series([0.6, 1.1, 2.6, 2.6, 4.0, 4.2, 4.8, 5.3, 5.5, 5.7, 5.8, 6.6, 8.4, 8.6, 9.5])

# Define an f-value function to calculate normalised f-values
def fvalue(var):
	i = np.arange(1, len(var) + 1)
	return (i - 0.5) / len(var)

fig, ax = plt.subplots(1)
ax.scatter(fvalue(x), x)
ax.set_ylim(0, 10)
ax.set_xlabel('$f')
ax.set_ylabel('$x')
ax.set_title('Quantile plot of the distribution of a random variable')
plt.show()
```

Output:

![[quantile_plot_statistics.png]]

- f quantile of a distribution is a number, q, where approximately a fraction f of the values of the distribution is less than or equal to q

$$ f_i = (i-0.5)/N $$

where
- i indexes the data values sorted in ascending order, starting from 1
- N is the number of data items

## Quantile-Quantile plot

- used to visualise more than one dataset
- used to compare empirical data against theoratical distributions

``` python
df = pd.DataFrame({'x': [0.6, 1.1, 2.6, 2.6, 4.0, 4.2, 4.8, 5.3, 5.5, 5.7, 5.8, 6.6, 8.4, 8.6, 9.5]})

# Specify the quantiles, avoiding 0 and 1 which correspond to infinity in the theoratical normal distribution
quantiles = np.linspace(0.001, 0.999, len(df))

# stats.norm.ppf = normal distribution percent point function,
# arguments are mean and standard deviation. This is the inverse of 
# the cumulative distribution function `stats.norm.cdf()`, in other
# words quantiles
df['norm_quantiles'] = stats.norm.ppf(quantiles, 0, 1)
ax = df.plot.scatter('norm_quantiles', 'x')
ax.set_ylim(0, 10)
ax.set_xlabel('Standard normal quantiles')
ax.set_ylabel('$x$')
ax.set_title('''Q-Q plot of the distribution of a random variable against the quantiles of the normal distribution''')

# Plot a reference line
ax.plot([0,1], [0,1], transform=ax.transAxes, color='lightgrey', zorder=-1)
plt.show()
```

Output:

![[quantile_quantile_plot.png]]



# Standard deviation

- measures of spread
- quantifies variation or dispersion from the mean
- high values = data is more spread out
- low values = data is closer to the mean

## Population standard deviation

$$ \sigma = \sqrt{\frac{\sum_{i=1}^N(x_i - \bar x)^2}{N}} $$

- where

$$ \text{- } \sigma \text{ is the population standard deviation} $$
$$ \text{- N is the population size} $$
$$ \text{- } x_i \text{ is the value of data item i} $$
$$ \text{- } \bar x \text{ is the population} $$

- average distance that data points are spread from the mean

## Corrected sample standard deviation

$$ s = \sqrt{\frac{\sum_{i=1}^N(x_i - \bar x)^2}{N-1}} $$

- where

$$ \text{- s is the population standard deviation} $$
$$ \text{- N is the sample size} $$
$$ \text{- } x_i \text{ is the value of data item i} $$
$$ \text{- } \bar x \text{ is the mean of the sample} $$

- average distance that data points are spread from the mean
- there will still be some bias for small samples, but as sample size increase, bias decreases

# Normal distribution

![[normal_distribution_statistics.png]]

# Box plot vs Density plot

![[box_plot_vs_density_plot.png]]

# z-score normalisation

- std can be used to calculate z-values or z-scores
- represent data values in terms of std from the mean
- useful for comparing variables measured on different scales

$$ z = \frac{x_i - \bar x}{s} $$

where
$$ \text{- } x_i \text{ is the value of data item i} $$
$$ \text{- } \bar x \text{ is the mean of the population} $$
$$ \text{- s is the standard deviation} $$

# Shape

- skewness
	- quantifies the asymmetry of a distribution
- kurtosis
	- quantifies the 'tailedness'
	- the shape of a distribution's tails in relation to its overall shape

## Skewness

### Non-skewed

``` python
# Seed the random number generator
np.random.seed(8)

s = pd.Series(np.random.normal(0, 1, 300))
ax = s.plot.hist(density=True)

quantiles = np.linspace(stats.norm.ppf(0.001), 
					   stats.norm.ppf(0.999), 100)
ax.plot(quantiles, stats.norm.pdf(quantiles))

ax.annotate(f'skew = {s.skew().round(2)}', xy = (-3, 0.35))
ax.set_xlabel('standard normal quantiles')
ax.set_ylabel('density')
ax.set_title('Non-skewed normally distributed random data')
plt.show()
```

Output:

![[non-skewed_statistics.png]]

### Left-skewed

``` python
# Seed the random number generator
np.random.seed(8)

s = pd.Series(np.random.normal(0, 1, 300))
s = s.append(pd.Series(np.random.normal(-2.5, 2.1, 100)), ignore_index=True)
ax = s.plot.hist(density=True)

quantiles = np.linspace(stats.norm.ppf(0.001), 
					   stats.norm.ppf(0.999), 100)
ax.plot(quantiles, stats.norm.pdf(quantiles))

ax.annotate(f'skew = {s.skew().round(2)}', xy = (-7.5, 0.35))
ax.set_xlabel('standard normal quantiles')
ax.set_ylabel('density')
ax.set_title('A negative (left) skewed random distribution')
plt.show()
```

Output:

![[left_skewed_statistics.png]]

### Right-skewed

``` python
# Seed the random number generator
np.random.seed(8)

s = pd.Series(np.random.normal(0, 1, 300))
s = s.append(pd.Series(np.random.normal(-2.5, 2.1, 100)), ignore_index=True)
s = -s
ax = s.plot.hist(density=True)

quantiles = np.linspace(stats.norm.ppf(0.001), 
					   stats.norm.ppf(0.999), 100)
ax.plot(quantiles, stats.norm.pdf(quantiles))

ax.annotate(f'skew = {s.skew().round(2)}', xy = (5, 0.35))
ax.set_xlabel('standard normal quantiles')
ax.set_ylabel('density')
ax.set_title('A positive (right) skewed random distribution')
plt.show()
```

Output:

![[right_skewed_statistics.png]]

## Kurtosis

### High kurtosis distribution

``` python
# Seed the random number generator
np.random.seed(8)

s = pd.Series(np.random.normal(0, 1, 300))
ax = s.plot.hist(density=True)

x = np.linspace(stats.norm.ppf(0.001), 
			   stats.norm.ppf(0.999), 100)
ax.plot(x, stats.norm.pdf(x))

ax.annotate(f'kurtosis = {s.kurtosis().round(2)}', xy = (-3, 0.35))
ax.set_ylabel('density')
ax.set_title('High kurtosis distribution')
plt.show()
```

![[high_kurtosis_statistics.png]]

### Leptokurtic distribution

- more outliers compared with the normal distribution

``` python
# Seed the random number generator
np.random.seed(8)

s = pd.Series(np.random.normal(0, 1, 300))
s.append(pd.Series(np.random.normal(0, 2, 300)), ignore_index=True)
ax = s.plot.hist(density=True)

x = np.linspace(stats.norm.ppf(0.001), 
			   stats.norm.ppf(0.999), 100)
ax.plot(x, stats.norm.pdf(x))

ax.annotate(f'kurtosis = {s.kurtosis().round(2)}', xy = (-5, 0.35))
ax.set_ylabel('density')
ax.set_xlim(-6, 6)
ax.set_title('Leptokurtic distribution')
plt.show()
```

![[leptokurtic_distribution_statistics.png]]

### Platykurtic distribution

- similar probability across its range

``` python
# Seed the random number generator
np.random.seed(5)

s = pd.Series(np.random.uniform(-2, 2, 300))
ax = s.plot.hist(density=True)

x = np.linspace(stats.norm.ppf(0.001), 
			   stats.norm.ppf(0.999), 100)
ax.plot(x, stats.norm.pdf(x))

ax.annotate(f'kurtosis = {s.kurtosis().round(2)}', xy = (-3, 0.35))
ax.set_ylabel('density')
ax.set_title('Platykurtic distribution')
plt.show()
```

![[platykurtic_distribution_statistics.png]]

# Nominal variables

## Univariate analysis (one variable)

- descriptive statistics of a single variable

## Multivariate analysis (two or more variable)

- descriptive statistics of a single variable in relation to categorical variables
- relationships between numerical variables
	- correlation

Coursera covered some simple data processing as well as code for grouped bar chart, stacked bar chart, normalised stack bar chart and heatmaps. Video link: [4.201 Nominal variables](https://www.coursera.org/learn/uol-cm3005-data-science/lecture/x4Z0u/4-201-nominal-variables)

# Ordinal variables

Coursera covered about normalised stacked bar, heat map and diverging stacked bar chart (Likert plot). It also showed some data processing to make generating the visualisation easier as well as the code for it. Video link: [4.203 ordinal variables](https://www.coursera.org/learn/uol-cm3005-data-science/lecture/mNFeh/4-203-ordinal-variables)

# Numerical and categorical data

## Cat plot

![[cat_plot_code.png]]

![[numerical_categorical_cat_plot.png]]

### With genders

![[cat_plot_genders_code.png]]

![[cat_plot_genders.png]]

## Box plot

![[box_plot_code.png]]

![[box_plot_statistics.png]]

### Seaborn version

![[box_plot_seaborn_code.png]]

![[box_plot_seaborn.png]]

## Numerical data

### Correlation coefficient

- a measure of how two variables change in a similar way
- relationship is generally assumed to be linear
- quantified on a scale -1 < r < 1

r = 1
- x and y are positively correlated

r = 0
- x and y are not correlated

r = -1
- x and y are negatively correlated

$$ r = \frac{\sum_{i=1}^n(X_i-\bar X)(Y_i-\bar Y)}{\sqrt{\sum_{i=1}^n(X_i-\bar X)^2}\sqrt{\sum_{i=1}^n (Y_i-\bar Y)^2}} $$

### Visualising correlation

![[visualising_correlation_code1.png]]

![[visualising_correlation_code2.png]]

![[visualising_correlation.png]]

#### Seaborn version

![[visualising_correlation_seaborn_code.png]]

![[visualising_correlation_seaborn.png]]

#### Regression estimate confidence bands

- the blue shaded area in the graph above is the region where a best fit regression line is 95% confidence within the region