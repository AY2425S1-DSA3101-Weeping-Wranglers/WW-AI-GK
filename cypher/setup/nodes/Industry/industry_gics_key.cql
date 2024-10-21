CREATE CONSTRAINT industry_gics_key IF NOT EXISTS
FOR (i:Industry) REQUIRE i.gics IS NODE KEY;