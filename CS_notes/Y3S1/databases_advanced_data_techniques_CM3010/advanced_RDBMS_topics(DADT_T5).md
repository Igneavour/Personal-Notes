---
tags: [clustered-indexing, non-clustered-indexing, b-tree, hash-table, normalisation, denormalisation]
aliases: [DADT T5, Databases and Advanced Data Techniques Topic 5]
---

# Query Efficiency

- Expensive operations
	- searching
	- sorting
	- copying data (reading & writing)

## using sorted tables (clustered indexing)

- as fast as tree indexes
- use no extra space
- can only sort in one order

## using indexes

- usually one of: 
	- B-tree
	- hash table
- may also be spatial/geometric
- may be held in memory
- may optimise for disk structure

### B-trees

- O(log n) in typical and worst case
- supports ranges
- supports (some) approximate searching

### Hash tables

- O(1) in typical case
- O(n) in unusual worst case
- Hash algorithm may be expensive
- no approximation
- no range-based retrieval

## Optimising actions

- indexes and sorted tables can save on copying, searching and sorting
- query strategy has a huge impact
	- order of operations
	- use of indexes
	- making fresh indexes
	- copying or reading data

## How to work out when to optimise

### Observing what operations would be performed on which tables in which order using <code>EXPLAIN</code> command 

``` SQL
EXPLAIN SELECT *
FROM MovieLocations, 
	Actors
WHERE Actor1=Name;
```

![[EXPLAIN_SQL1.png]]

![[EXPLAIN_SQL2.png]]

Total runtime required: ~60ms

#### creating non-clustered databases using indexes

``` SQL
CREATE INDEX ActorNames ON Actors(Name);
```

(for better explanation of this, refer to this article I found online myself) :

[How to use Indexing to Improve Database Queries](https://dataschool.com/sql-optimization/how-indexing-works/#:~:text=Indexing%20makes%20columns%20faster%20to,row%20until%20it%20finds%20it.)

![[EXPLAIN_SQL_INDEXED1.png]]

![[EXPLAIN_SQL_INDEXED2.png]]

Total runtime required: ~5ms

## Optimising join-based queries

![[join_based_queries_example.png]]

# Denormalisation

## Normalisation

- can reduce disk reads
- can reduce integrity checks
- reduces storage use
However:
- increases use of joins (expensive operations)

## Denormalisation

- merge tables to reduce joins
- effectively caches a joined SELECT
- reduces use of joins
- MAY sometimes be faster
- <b>risky</b> for very dynamic data

## Examples of denormalisation using existing tables

``` SQL
CREATE TABLE MovieActors(PRIMARY KEY (Title, Year, Location))
AS
SELECT Title, Year, Location, Name, DoB, Gender 
FROM MovieLocations
LEFT JOIN
Actors
ON Actor1=Name;
```

## Alternative cache (Dynamic)

``` SQL
CREATE VIEW SFMoviesAndActors
AS
SELECT Title, Year, Location, Name, DoB, Gender
FROM MovieLocations
LEFT JOIN
Actors
ON Actor1=Name;
```

## Caching views 

- SQL standard: SNAPSHOT
- AKA Embodied or Materialised Views
- Can update when data changes
- Not implemented in MySQL

## Conclusion

- databases are fast
- key-based joins are optimized
- evaluate speed and efficiency before trying to optimize