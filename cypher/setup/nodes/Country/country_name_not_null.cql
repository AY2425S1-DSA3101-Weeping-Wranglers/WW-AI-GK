CREATE CONSTRAINT country_name_not_null IF NOT EXISTS
FOR (c:Country) REQUIRE c.name IS NOT NULL;