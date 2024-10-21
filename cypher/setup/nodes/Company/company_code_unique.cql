CREATE CONSTRAINT company_code_unique IF NOT EXISTS
FOR (c:Company) REQUIRE c.ticker_code IS UNIQUE;