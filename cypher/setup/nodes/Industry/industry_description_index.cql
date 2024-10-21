CREATE VECTOR INDEX industry_description_index IF NOT EXISTS
FOR (i:Industry)
ON i.description
OPTIONS { indexConfig: {
 `vector.dimensions`: 2048,
 `vector.quantization.enabled`: false
}};