MATCH (c:Country {iso3: $iso3})
SET
    c.population = $population,
    c.gdp = $gdp,
    c.pv = $pv,
    c.corporate_tax_rate = $corporate_tax_rate;