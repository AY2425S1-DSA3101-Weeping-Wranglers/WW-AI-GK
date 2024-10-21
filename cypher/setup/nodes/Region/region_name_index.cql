CREATE FULLTEXT INDEX region_name_index IF NOT EXISTS
FOR (r:Region) ON EACH [r.name];