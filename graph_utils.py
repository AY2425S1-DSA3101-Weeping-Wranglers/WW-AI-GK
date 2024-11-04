import dotenv
import os
from neo4j import GraphDatabase, EagerResult
from neo4j.exceptions import ServiceUnavailable

load_status = dotenv.load_dotenv()
if load_status is False:
    raise RuntimeError('Environment variables not loaded.')

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

try:
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
except ServiceUnavailable:
    URI = URI.replace("neo4j+s", "neo4j+ssc")
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()

print("Connection to graph database established.")

def execute_query(query: str) -> EagerResult:
    '''
    Executes a query without any parameters
    '''
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        result = driver.execute_query(query)
    
    return result

def execute_query_with_params(query: str,
                              *param_dicts: dict[str, str]) -> list[EagerResult]:
    '''
    Executes a given query with each param_dict in param_dicts.
    Transaction based - All queries must be successful for changes to be committed.
    '''
    results = []
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database="neo4j") as session:
            with session.begin_transaction() as tx:
                for param_dict in param_dicts:
                    result = tx.run(query, param_dict)
                    results.append(result.to_eager_result())
                tx.commit()
    return results

def reset_graph():
    '''
    Deletes all nodes and relationshipss
    '''
    execute_query("MATCH (n) DETACH DELETE n")

def reset_constraints():
    '''
    Deletes all constraints and indexes
    '''
    execute_query("CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *")