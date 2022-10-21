---
tags: [relational-database, relational-model, entity-relationship-model, SQL, cardinality, integrity-constraint]
aliases: [DADT T2, Databases and Advanced Data Techniques Topic 2]
---

# What is a relational database?

- Relational databases implement the relational <b>model</b>

## Model

- a way of abstracting complex real-world data into structures
- the way we represent data itself in the abstract

## Relational

> Relational model != Entity-relationship model

- A relation is approximately a table or, a definition of a table and all the values stored in it

## Rules

- All operations use the relational model
- All data is represented and accessed as relations
- Table and database structure is accessed and altered as relations
- The system is unaffected by its implementations:
	- if the hardware changes
	- if the OS changes
	- if the disks are replaced
	- if data is distributed

# Entity-Relationship model

## Entity

- A thing we want to model
- Can be uniquely identified

### Drawing Entity

![[entity.png]]

## Attribute

- Information that describes an aspect of an entity

### Drawing Attribute

![[attribute.png]]

## Entity with attribute

![[entity_with_attribute.png]]

## Relationship

- A connection or dependency between two entities

### Drawing Relationship

![[relationship.png]]

# SQL

Basic explanations on commonly used statements for data manipulation and data definition in Coursera week 3 and 4, can also just refer online.

week 3

Data Definition
- CREATE
- DROP
- TRUNCATE
- ALTER

Data Manipulation
- SELECT
- INSERT
- UPDATE
- DELETE

week 4
- CROSS-JOIN
- INNER-JOIN
- LEFT-JOIN

# Cardinality

Cardinality is concerned with how many rows in each of the tables that participate in a join match with how many rows in the other table.

## Importance of Cardinality
- Cardinality affects implementation
	- In the form of a foreign key for example

one-to-one relationship: primary key

one-to-many relationship: primary key + foreign key

many-to-many relationship: needs to first be converted to one-to-many, many-to-one as this cannot directly be modifiable in the relational model. Adding a new entity will help solve this issue.

# Issues that can arise in database

- NULL values due to forgetting to input

![[null_values.png]]

- Spelling or wrong input

![[misspelt_values.png]]

- Invalid or out-of-range input

![[meaningless_values.png]]

- Contradictory or inconsistent values

![[inconsistent_values.png]]

- Incompatible values

![[incompatible_values.png]]

- Change that cause 'orphan' record

![[orphaned_values.png]]

# Integrity constraints

- Use foreign keys to preserve integrity of the database
	- Use foreign key <b>rules</b> to enforce behaviour

![[foreign_key_rules.png]]

- Use CHECK column constraint to ensure values of a field are valid
- Use primary keys to guarantee uniqueness and avoid repeating information
- Avoid storing calculated values
- Remove functional dependencies