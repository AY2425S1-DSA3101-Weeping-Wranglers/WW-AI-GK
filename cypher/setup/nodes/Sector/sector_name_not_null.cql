CREATE CONSTRAINT sector_name_not_null IF NOT EXISTS
FOR (s:Sector) REQUIRE s.name IS NOT NULL;