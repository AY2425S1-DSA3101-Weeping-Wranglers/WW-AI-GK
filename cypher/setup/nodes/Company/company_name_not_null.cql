CREATE CONSTRAINT company_name_not_null IF NOT EXISTS
FOR (c:Company) REQUIRE c.name IS NOT NULL;