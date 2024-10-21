MATCH (c:Country {iso3: $iso3})
SET c.aliases = 
    CASE
        WHEN c.aliases IS NULL THEN [$alias]
        WHEN NOT $alias IN c.aliases THEN c.aliases + $alias
        ELSE c.aliases
    END;