from pyspark.sql import SparkSession
import pandas as pd
import json
import getpass
import requests

# Initialize Spark session for distributed processing
spark = SparkSession.builder.appName("ScaleEntityRelationshipExtraction").getOrCreate()

# Global variables
TOKEN = getpass('Enter token: ')
FIELDS = "entities,facts"
HOST = "nl.diffbot.com"
MIN_SALIENCE = 0.5

# --- Data Ingestion ---
def load_data(file_path):
    """Load ECM data from a distributed storage source."""
    return spark.read.text(file_path)

# --- Entity Extraction and Classification ---
def extract_entities(api_response):
    """Extract and filter entities from API response based on salience."""
    entities = api_response.get("entities", [])
    salient_entities = [ent for ent in entities if ent["salience"] >= MIN_SALIENCE]

    # Classify entities based on their types
    classified_entities = [
        {
            "name": ent["name"],
            "salience": ent["salience"],
            "Labels": classify_entity(ent)  # Classify based on type
        }
        for ent in salient_entities
    ]
    
    return pd.DataFrame(classified_entities)

def classify_entity(entity):
    """Classify entity type based on its attributes."""
    entity_types = [etype["name"] for etype in entity.get("allTypes", [])]
    if "organization" in entity_types:
        return 'company'
    elif "industry" in entity_types:
        return 'industry'
    elif "country" in entity_types:
        return 'country'
    elif "location" in entity_types:
        return 'location'
    elif "product" in entity_types:
        return 'product'
    return entity_types[0] if entity_types else "unknown"

# --- Relationship Extraction ---
def extract_relationships(api_response):
    """Extract relationships from API response data."""
    relationships = api_response.get("relationships", [])
    extracted_relationships = [
        {
            "subject": rel["subject"],
            "object": rel["object"],
            "type": rel["type"]
        }
        for rel in relationships
    ]
    return pd.DataFrame(extracted_relationships)

# --- API Request Handling ---

def get_request(payload):
  """Make a call to Diffbot's Natural Language API to retrieve entity and relationship data."""
  res = requests.post("https://{}/v1/?fields={}&token={}".format(HOST, FIELDS, TOKEN), json=payload)
  ret = None
  try:
    ret = res.json()
  except:
    print("Bad response: " + res.text)
    print(res.status_code)
    print(res.headers)
  return ret

# --- Distributed Processing with Spark ---
def process_entities(partition):
    """Process entity extraction in parallel across partitions."""
    entities = []
    for row in partition:
        api_response = get_request(row["value"])
        entities_df = extract_entities(api_response)
        entities.append(entities_df)
    return pd.concat(entities) if entities else pd.DataFrame()

def process_relationships(partition):
    """Process relationship extraction in parallel across partitions."""
    relationships = []
    for row in partition:
        api_response = get_request(row["value"])
        rels_df = extract_relationships(api_response)
        relationships.append(rels_df)
    return pd.concat(relationships) if relationships else pd.DataFrame()

# --- Pipeline Execution ---
def main(input_path, output_path):
    """Main function to run the distributed Entity-Relationship Extraction pipeline."""
    # Load data from the specified input path
    ecm_data = load_data(input_path)

    # Distributed Entity Extraction
    entities_rdd = ecm_data.rdd.mapPartitions(process_entities)
    entities_df = spark.createDataFrame(entities_rdd).cache()

    # Distributed Relationship Extraction
    relationships_rdd = ecm_data.rdd.mapPartitions(process_relationships)
    relationships_df = spark.createDataFrame(relationships_rdd).cache()

    # Save results to JSON format
    entities_json = entities_df.toJSON().collect()
    relationships_json = relationships_df.toJSON().collect()

    # Aggregate and store in JSON schema
    final_output = {
        "entities": [json.loads(ent) for ent in entities_json],
        "relationships": [json.loads(rel) for rel in relationships_json]
    }

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Entity-Relationship Extraction completed. Results saved to {output_path}")

# --- Run Script ---
if __name__ == "__main__":
    # Input and output paths for ECM data and results
    input_path = "s3://your-bucket/ECM_data.txt"  # Modify this to your input data path
    output_path = "output/EntityRelationshipExtractionResults.json"  # Modify as desired

    main(input_path, output_path)