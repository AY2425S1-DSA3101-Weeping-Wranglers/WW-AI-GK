CREATE CONSTRAINT industry_description_not_null IF NOT EXISTS
FOR (i:Industry) REQUIRE i.description IS NOT NULL;