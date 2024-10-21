CREATE CONSTRAINT industry_name_unique IF NOT EXISTS
FOR (i:Industry) REQUIRE i.name IS UNIQUE;