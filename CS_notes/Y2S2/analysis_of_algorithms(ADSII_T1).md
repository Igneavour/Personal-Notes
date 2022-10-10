---
tags: [algorithms, ADSII, Algorithms-&-Data-Structures-II, algorithms-analysis, algorithm-growth, algorithm-runtime]
alias: [Algorithms & Data Structure II Topic 1 revision notes, ADSII T1]
---

# Empirical Measurement

| Advantages               | Drawbacks                 |
| ------------------------ | ------------------------- |
| Real/exact result        | Machine-dependent results |
| No need for calculations | Implementation effort     |

# Theoratical Measurement

| Advantages               | Drawbacks           |
| ------------------------ | ------------------- |
| Universal results        | Approximate results |
| No implementation effort | Calculations effort |

# Running time assumptions

- A single CPU
- One simple operation = 1 time unit
- Loops and functions are <b>NOT</b> simple operations
- No memory hierarchy --> any memory access = 1 time unit

# Memory usage assumption

- A simple variable uses 1 memory position
- An array of N elements uses N memory positions

# Counting time and space units example

```
function F1(a,b,c)
max = a ----------------> 2 time units (memory read + memory write)
if (b > max) -----------> 4 time units (2 memory read + 1 comparison + 1 if statement)
	max = b ------------> 2 time units
if (c > max) -----------> 4 time units
	max = c ------------> 2 time units
return max -------------> 2 time units (1 memory read + 1 return statement)
```

1 variable declared (max) = 1 space unit

```
function F2(A, N, x)
for 0 <= i < N -------> 4*(N+1) + 3N
	if (A[i] == x) ---> 5N
		return i ----- 1
return -1 ___________/		
```

For-loop explanation:
- N+1 because we assume integer does not exist in array, multiplied by 4 for the 4 time units by the operations within the inner if-statement to check whether i < N (1 if, 2 memory read for i and N, 1 operation <). +3N is for updating i or the control variable (i = i + 1).
- 5N (3 memory reads (x, i, A[i]), 1 numerical operator (\==), 1 if statement)
- Last 1 is either return 1 or -1.

1 space unit for i in the for-loop

# Growth of running time

* Use generic constants e.g: T(N) = C<sub>1</sub>N + C<sub>2</sub>
* Ignore constants and lower-order terms

# Analysis of algorithm run-time

```
function SumDiag(A)
sum <-- 0
N <-- length(A[0]) ------------> C1N + C2
for 0 <= i < N ----------------> C3N + C4
	sum <-- sum + A[i,i] ------> C5N
return sum --------------------> C6
```

T(N) = C<sub>7</sub>N + C<sub>8</sub> 

# Worst case analysis

For an input of size N, you analyse the case that takes the maxiumum number of instructions.

# Best case analysis

For an input of size N, you analyse the case that takes the minimum number of instructions.

# Asymptotic Analysis

## Big-O notation

> T(n) is O(g(N)) if:
> c*g(N) >= T(N) for all N >= n<sub>0</sub>
> c and n<sub>0</sub> must be positive constants.

- It is fine if there are values for which c*g(N) < T(N) as long as ** for large values of N ** , c\*g(N) is an upper bound for T(N). 

E.g:
- T(N) is O(N<sup>3</sup>)
- T(N) is O(N<sup>4</sup>)
- T(N) is O(2<sup>N</sup>)

## Omega notation

> T(n) is Ω(g(N)) if:
> c*g(N) <= T(N) for all N >= n<sub>0</sub>
> c and n<sub>0</sub> must be positive constants.

- It is fine if there are values for which c*g(N) < T(N) as long as ** for large values of N ** , c\*g(N) is an upper bound for T(N). 

E.g:
- T(N<sup>3</sup>) is Ω(g(N))
- T(N<sup>4</sup>) is Ω(g(N))
- T(2<sup>N</sup>) is Ω(g(N))

## Theta notation

> T(n) is Θ(g(N)) if:
> c<sub>1</sub>\*g(N) <= T(N) for all N >= n<sub>0</sub>
> c<sub>2</sub>\*g(N) >= T(N) for all N >= n<sub>0</sub>

f(N) is Θ(g(N)) if there exist positive constants c<sub>1</sub>, c<sub>2</sub> and n<sub>0</sub> such that:

c<sub>1</sub>g(N) <= f(N) <= c<sub>2</sub>g(N) for all n>= n<sub>0</sub> 