MATCH
    (c:Country{iso3: $iso3}),
    (r:Region{m49: $m49})
MERGE (c)-[:IS_IN]->(r);