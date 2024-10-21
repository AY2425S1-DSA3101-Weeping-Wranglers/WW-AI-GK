CREATE CONSTRAINT industry_name_not_null IF NOT EXISTS
FOR (i:Industry) REQUIRE i.name IS NOT NULL;