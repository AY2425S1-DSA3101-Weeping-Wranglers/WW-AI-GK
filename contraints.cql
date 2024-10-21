// Remove any existing self-loop relationships
MATCH (c:Company)-[r:PARTNERS_WITH]->(c)
DELETE r;

// Remove any existing self-loop relationships
MATCH (c:Company)-[r:SUBSIDIARY_OF]->(c)
DELETE r;

// Remove any existing self-loop relationships
MATCH (c:Company)-[r:COMPETES_WITH]->(c)
DELETE r;