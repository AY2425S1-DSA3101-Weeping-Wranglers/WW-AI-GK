CREATE CONSTRAINT region_name_not_null IF NOT EXISTS
FOR (r:Region) REQUIRE r.name IS NOT NULL;