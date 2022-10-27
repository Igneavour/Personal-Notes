---
tags:
aliases:
---

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