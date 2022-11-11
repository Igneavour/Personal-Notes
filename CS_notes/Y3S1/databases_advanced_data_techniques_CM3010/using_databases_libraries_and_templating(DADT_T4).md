---
tags: [updates, security-concerns, SQL-injection, new-id, data-exchange, database-syntax]
aliases: [DADT T4, Databases and Advanced Data Techniques Topic 4]
---

# Speaking to databases

## SELECT

``` SQL
SELECT * FROM Planets WHERE diameter > 5;
```

Approximate matching:

``` SQL
SELECT * FROM Planets WHERE name LIKE '%er';
```

## UPDATE

``` SQL
UPDATE Planets SET name='Mars' WHERE name='Mers';
```

## INSERT

``` SQL
INSERT INTO Planets(name, diameter) VALUES('Gethen', 9483)
```

Insert results of a query into a table:

``` SQL
INSERT INTO Planets(name) SELECT DISTINCT name FROM Moons;
```

## DELETE

``` SQL
DELETE FROM Planets WHERE diameter < 2500;
```

## DROP

> Deletes the entire table itself

``` SQL
DROP TABLE Planets;
```

## TRUNCATE

> Removes all entries but keep database structure or table

``` SQL
TRUNCATE TABLE Planets;
```

## CREATE

``` SQL
CREATE TABLE Planets(PlanetName CHAR(8), 
					 DayLength INT, 
					 YearLength INT, 
					 PRIMARY KEY(PlanetName));
```

## ALTER

``` SQL
ALTER TABLE Planets ADD COLUMN Diameter INT;
```

# Data exchange

Exporting data:

``` SQL
SELECT * FROM Planets INTO OUTFILE 'Planets.txt';
```

Importing data:

``` SQL
LOAD DATA INFILE 'Planets.txt' INTO TABLE Planets;
```

# Connecting to an SQL RBDMS

## Database libraries

- Create a persistent connection
- Send commands
- Receive and structure response

## Creating a connection

``` 
conn <-- newConnection(host, username, password, database)
conn.connect()
```

## Sending commands

```
resource <-- conn.execute(query)
resource.fetchData()
		or
result <-- conn.query(query)
```

## Receive response

- Result will usually be iterable
- Each row may be an object or an array



# Making updates

``` javascript
const addActor = `
INSERT into Actors VALUES(
"Richard, Gere", "Male", Richard Gere", "1949-08-31");
`;
connect.query(addActor);
```

Programmatically
``` Javascript
const addActor = `
INSERT into Actors VALUES(
"`+actor.name
+'", "'+actor.gender
+'", "'+actor.name
+'", "'+actor.birthDate
+'");`;
```

# Security concerns

SQL injection is possible for any user or HTTP-supplied data
- Control user privileges
- Escape user input
- Restrict possible operations

## Making secure updates

Letting the system do the escaping for us

``` Javascript
const addActor = `
INSERT into Actors
VALUES
(?, ?, ?, ?);`;
connect.query(addActor, [actor.name, actor.gender, actor.name, actor.birthDate]);
```

Stored procedures:

``` SQL
delimiter //
CREATE PROCEDURE addActor
(IN name, gender, dob)
BEGIN
	INSERT INTO Actors
	VALUES(name, gender, name, dob);
END //
delimiter;
```

``` Javascript
connect.query(`
	CALL addActor(
		"Richard Gere", "male", 
		"1949-08-31");
		`);
```

``` SQL
GRANT EXECUTE ON addActor TO webUser;
```

# New IDs

- Auto-increment IDs are useful
- INSERT commands will result in a new ID being generated
- SELECT last_insert_id();
- In node:
``` Javascript
conn.query(insertCommand,
  function(err, res, cols){
	  res.insertId
	  });
```

# Python for databases

``` python
cursor = conn.cursor()
cursor.execute(query)
for row in cursor:
	print(row[1])
```
OR
``` python
cursor = conn.cursor()
cursor.execute(query)
result = cursor.fetchall()
```

When we want to access by name rather than index:

``` python
cursor = conn.cursor(dictionary="true")
cursor.execute(query)
for row in cursor:
	print(row['diameter'])
```