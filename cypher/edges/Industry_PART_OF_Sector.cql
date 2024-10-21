MATCH
    (i:Industry{gics: $industry_gics}),
    (s:Sector{gics: $sector_gics})
MERGE (i)-[:PART_OF]->(s);