CREATE CONSTRAINT country_name_unique IF NOT EXISTS
FOR (c:Country) REQUIRE c.name IS UNIQUE;