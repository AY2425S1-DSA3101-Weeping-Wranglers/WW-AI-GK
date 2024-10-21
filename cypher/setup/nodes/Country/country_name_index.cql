CREATE FULLTEXT INDEX country_name_index IF NOT EXISTS
FOR (c:Country) ON EACH [c.aliases];