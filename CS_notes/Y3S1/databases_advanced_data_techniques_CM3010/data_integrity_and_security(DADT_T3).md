---
tags: [sources-of-error, integrity,consistency, anomaly, non-loss-decomposition, functional-dependency, normalisation, heath's-theorem, boyce-codd-normal-form, transitive-dependency, multi-valued-dependency, ACID, transactions, serialisability, roles, user-policy, SQL-injection]
aliases: [DADT T3, Database and Advanced Data Techniques Topic 3]
---

# Reading resources for this topic

1. [Database Design and Relational Theory chapter 4 FDs and BCNF](https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=5997273)

# Sources of error

- bad data
	- automated integrity checks but does not rid of human factor
- poor application logic
	- risk reduction - normalisation
- failed database operations
	- saving database state - snapshots
	- blocks of commands - transactions
- malicious user activity
	- user privileges and security

# Integrity

Specify PRIMARY or UNIQUE KEY
- ensures every row is identifiable
- will throw an error if any change will create a duplicate

Specify FOREIGN KEY
- ensures every reference is maintainable
- will throw an error if any change will create a reference to a non-existent key
- will propagate changes to parent table

Specify CHECK
- ensures column data is valid

<p style="font-size:20px;">Integrity checks will not check for truth, it will only check the database for consistency and logical integrity</p>

# Consistency and anomalies

## Bad table example 1

![[inconsistent_anomalies.png]]

- inconsistent naming of the same filming locations
- if we were to change one, the rest would not be affected

## Bad table example 2

![[inconsistent_table_example2.png]]

- should have multiple actors, but we do not know the numbers and thus, inconsistent number of columns needed
- naming of filming locations not consistent again
- multiple distributors and brand name changes, distributors also appear multiple times in different films

<p style="font-size:25px;">Here is a possible entity-relationship diagram to normalise the database</p>

![[entity_relationship_diagram_for_example2.png]]

- many-to-many hinting at possibly needing extra tables
- distributors, actors and distributors can have 1 or 2 or many of them

# Non-loss decomposition

> A decomposition of a single relation into two or more separate relations such that a join on the separate relations reconstructs the original

# Functional dependency

> An attribute Y is said to be functionally dependent  on another attribute X in the same relation if for any legal value of X there is exactly one associated value of Y

Going back to the previous example of the films [[data_integrity_and_security(DADT_T3)#Bad table example 2|here]], adding a Film_ID will ensure the rows are unique

Knowing the Film_ID will ensure that you know:
- Title
- Year
- Distributor
- Director
- Actor

Other dependies relationship:

Co_ID + year --> co_name

Person_ID --> person_name

Film_ID + ActorNo --> Person_ID

# Normalisation and the normal forms

## First normal form (1NF)

> The table is a relation. All of its attributes are scalar values (no columns contains an array or an object)

Non-normal form

![[example1_non_normal.png]]

Normalised table

![[example1_normalised.png]]

## Heath's Theorem

> A relation with attribute A, B and C with a functional dependency A --> B is equal to the join of {A,B} and {A,C}

## Second normal form (2NF)

> Table is in 1NF. Every non-key attribute is irreducibly dependent on the primary key

1NF table

![[example2_non_normal.png]]

2NF table

![[example2_normalised.png]]

## Third normal form (3NF)

> Table is 2NF. Every non-key attribute is non-transitively (directly) dependent on the primary key

### Transitive dependency

> A, B, C are attributes or sets of in a relation, with functional dependencies: 
> A --> B, 
> B --> C, 
> A --> C

2NF table

![[example3_non_normal.png]]

![[example3_normalised.png]]

## Boyce-Codd normal form (BCNF)

> Table is 3NF. All non-trivial functional dependencies depend on a superkey

## Fourth normal form (4NF)

> Table is in 3NF. For every Multi-Valued Dependency A --> B, A is a candidate key

### Multi-valued dependency

> A and B are two non-overlapping sets of attributes in a relation. There is a multi-valued dependency if the set of values for B depends only on the values of A
> A --> B is similar to a functional dependency, but with multiple optons for B

3NF 

![[example4_non_normal.png]]

4NF

![[example4_normalised.png]]

# Guaranteeing a DBMS against errors

## ACID properties

ACID can be satisfied by a combination of:
- Transactions
- Locking strategy
- Failure recovery strategy

- Atomicity - Groups of operations are all performed or none are 
- Consistency - The database is never in an inconsistent state as a result of groups or operations being processed
- Isolation - If two or more groups of operations affect the same data, they can only be performed one at a time
- Durability - A completed operation or group of operations will have its changes physically committed 

### Isolation

- During a block of operations, restrict access to data that be affected by any operation in the block

### Atomicity

- If an operation in a block fails, ROLLBACK to the state immediately before the block was started

### Durability

- Initial and final states should be recorded reliably

### Consistency

- Restrict access to intermediate states of the database. Only store initial and final states

## Transactions

``` SQL
START TRANSACTION;
SELECT ... # Intermediate states
INSERT ... # Intermediate states
UPDATE ... # Intermediate states
COMMIT;
```

If an error occurs in between:

``` SQL
START TRANSACTION;
SELECT ... # Intermediate states
INSERT ... # Intermediate states
UPDATE ... # Intermediate states
ROLLBACK;
```

## In practice

- Data Definition Language can cause problems
- Checkpoints may not be as frequent as transactions
- Table locking is not absolute (locking access to intermediate steps)

## Inconsistent analysis

> Two transaction access the same data. One has multiple queries which give inconsistent information

## Serialisability

> Interleaved execution of transactions is serializable if it produces the same result as a non-interleaved execution of the same transactions

- Operations on different data are serializable
- Operations which are all SELECTs are serializable

# Malicious and accidental damage

- SQL injection puts malicious code into normal operations
- Malicious agent gains direct access to the database
- User error
- Confidential data shared inappropriately

# Security and user policies with SQL

## Users in SQL

- Create, edit users
- Create, edit, use databases
- Create, edit, use tables
- Create, edit, use data

- Define your user policy in advance
- Consider whether a user needs separate 'roles'

## User privilege commands

GRANT (ability to do stuff)

ON (bits of the database)

TO (someone)

### Example

``` SQL
GRANT SELECT
ON Planets,
	Moons
TO User1;
```

Creating a user account
``` SQL
CREATE USER User1
IDENTIFIED BY 'someusername'
PASSWORD EXPIRE;
```

Giving other users up to and including the same privileges that they have:
``` SQL
GRANT SELECT
ON Planets,
	Moons
TO User1
WITH GRANT OPTION;
```

Revoking user privileges:
``` SQL
REVOKE ALL
ON Planets,
	Moon
FROM User2;
```

## Roles

> Something like a group where they can be assigned to an individual user. Also has various privileges tagged onto the role

### Purpose of roles

- Minimise user privileges to reduce impact of error or malice