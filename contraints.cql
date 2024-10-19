
//Company Node
//ticker_code is unique
CREATE CONSTRAINT company_code_unique IF NOT EXISTS
FOR (c:Company) REQUIRE c.ticker_code IS UNIQUE;

//ticker_code is not null
CREATE CONSTRAINT company_code_not_null IF NOT EXISTS
FOR (c:Company) REQUIRE c.ticker_code IS NOT NULL

//name is not null
CREATE CONSTRAINT company_name_not_null IF NOT EXISTS
FOR (c:Company) REQUIRE c.name IS NOT NULL

//founded_year is not null
CREATE CONSTRAINT company_founded_year_not_null IF NOT EXISTS
FOR (c:Company) REQUIRE c.founded_year IS NOT NULL


//Remove any existing self-loop relationships
MATCH (c:Company)-[r:PARTNERS_WITH]->(c)
DELETE r;

//Remove any existing self-loop relationships
MATCH (c:Company)-[r:SUBSIDIARY_OF]->(c)
DELETE r;

//Remove any existing self-loop relationships
MATCH (c:Company)-[r:COMPETES_WITH]->(c)
DELETE r;


//Industry Node
//SICcode is unique
CREATE CONSTRAINT Industry_SICcode_unique IF NOT EXISTS
FOR (x:Industry) REQUIRE x.SICcode IS UNIQUE;

//SICcode is not null
CREATE CONSTRAINT Industry_SICcode_not_null IF NOT EXISTS
FOR (x:Industry) REQUIRE x.SICcode IS NOT NULL

//name is unique
CREATE CONSTRAINT Industry_name_unique IF NOT EXISTS
FOR (x:Industry) REQUIRE x.name IS UNIQUE;

//name is not null
CREATE CONSTRAINT Industry_name_not_null IF NOT EXISTS
FOR (x:Industry) REQUIRE x.name IS NOT NULL

//industry_group is not null
CREATE CONSTRAINT Industry_group_not_null IF NOT EXISTS
FOR (x:Industry) REQUIRE x.industry_group IS NOT NULL


//Region
//m49 is unique
CREATE CONSTRAINT Region_m49_unique IF NOT EXISTS
FOR (r:Region) REQUIRE r.m49 IS UNIQUE;

//m49 is not null
CREATE CONSTRAINT Region_m49_not_null IF NOT EXISTS
FOR (r:Region) REQUIRE r.m49 IS NOT NULL

////name is unique
CREATE CONSTRAINT Region_name_unique IF NOT EXISTS
FOR (r:Region) REQUIRE r.name IS UNIQUE;

//name is not null
CREATE CONSTRAINT Region_name_not_null IF NOT EXISTS
FOR (r:Region) REQUIRE r.name IS NOT NULL

//Country
//iso3 is unique
CREATE CONSTRAINT Country_iso3_unique IF NOT EXISTS
FOR (c:Country) REQUIRE r.iso3 IS UNIQUE;

//iso3 is not null
CREATE CONSTRAINT Country_iso3_not_null IF NOT EXISTS
FOR (c:Country) REQUIRE r.iso3 IS NOT NULL

//iso2 is unique
CREATE CONSTRAINT Country_iso2_unique IF NOT EXISTS
FOR (c:Country) REQUIRE r.iso2 IS UNIQUE;

//name is unique
CREATE CONSTRAINT Country_name_unique IF NOT EXISTS
FOR (c:Country) REQUIRE r.name IS UNIQUE;

//name is not null
CREATE CONSTRAINT Country_name_not_null IF NOT EXISTS
FOR (c:Country) REQUIRE r.name IS NOT NULL


