---
tags: [obsidian-guide, notetaking]
aliases: [notetaking guide, obsidian guide]
---

# Basic Markdown syntax

Most of the contents covered here are things that I believe may come in handy to me often. Any other things that are not covered pertaining to syntaxes can be found by going to the markdown guide website. I will not cover any basic syntax like bolding, italic or other complicated syntaxes that I would not use often. 

Link to markdown syntax website: <form action="https://www.markdownguide.org/basic-syntax/#code-1"><input type="submit" value="Markdown basic syntax"></form>

# YAML frontmatter

As seen from above, those 2 are features provided by obsidian. I can also add my own keys/attributes. Ensure that you add the 3 hyphens at the start and end to make it it readable by obsidian.

# Links

## Internal links

To create links between files in obsidian, simply add two square brackets like: 
[[Quick Obsidian Guide to Notetaking for self]]

To link to headings, add a hashtag afterwards like so:

[[Quick Obsidian Guide to Notetaking for self#Links]]

## External links

To create an external link, you put a link within the parenthesis and an optional square brackets for the link text like:

### With square brackets naming

[Google](https://google.com)

### Without square brackets naming

https://google.com

## Preview links

Adding an exclamation mark in front of a link (double square brackets) will show a preview of the link/image like so:

![[ayaya.jpg]]

OR

![Picture of Plains](https://mdg.imgix.net/assets/images/shiprock.jpg?auto=format&fit=clip&q=40&w=1080)

Pretty cool huh?

## Embedded website

<iframe src="https://www.youtube.com" style="width: 800px;height: 800px;"></iframe>

## Fancy links

This is a little over the top for just a simple markdown text document, but you could try making a link using some HTML form action like this:

<form action="https://google.com">
	<input type="submit" value="Google" />
</form>

## Placeholder links

In the event there is a note or basically something you want to link to that has not been created yet, you can make it a placeholder by simply creating a link with exclamation mark in front and just write the name of that file that you will create eventually like this:

![[Doesn't exist Yet]]

# Table

To create a table, you need to start off with a OR operator --> |  followed by TAB to initiate the table-building process. This will start you off with a header cell which by pressing TAB again will bring you to the other TAB cell. Once you are done making headers, pressing ENTER will move you one row down to create your first row of content cell. Below is an example of a table: 

| Command                            | Control                       |
| ---------------------------------- | ----------------------------- |
| Create another header/content cell | TAB (after writing something) |
| Move to new row                    | ENTER                              |

# Code block

A simple way to show a block of code is to add three backticks at the start/top and another 3 at the end/bottom. If you add an additional name of programming language, it will highlight syntax-specific keywords.

``` python
print("hello world")
```

OR indent each line with a TAB or 4 spaces

	<html>
	<body>
		<h1>hi</h1>
	</body>	
	</html>

If you want to just have a small code, you can just enclose it with backticks like 

``print("hello world")`` 

OR enclose with the code tag like

<code>print("hello world")</code>



# Block quotes

> Sample block quote

# Latex 

To write mathematical expressions or equations, use double $ sign like the example below. Any specific mathematical symbols not known shall be googled: 

$$ \frac{a}{b} $$