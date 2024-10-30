import os
import csv
import requests
from collections.abc import Iterable
import tempfile
import graph_utils
from collections import Counter, defaultdict

__ID_VALUES__ = {}

def extract_key_constraints() -> dict[str, str]:

    '''
    Extracts all key properties of nodes from graph
    '''

    records, _, _ = graph_utils.execute_query("""
SHOW KEY CONSTRAINTS
YIELD labelsOrTypes, properties
RETURN labelsOrTypes, properties""")
    result = {}
    for record in records:
        label = record[0][0]
        property = record[1][0]
        result[label] = property
    return result

def extract_relations() -> list[tuple[str, str, str]]:

    '''
    Extracts all relation types from graph
    '''

    records, _, _ = graph_utils.execute_query("""
MATCH (n)-[r]->(m)
RETURN DISTINCT
    LABELS(n),
    TYPE(r),
    LABELS(m)""")
    result = []
    for record in records:
        src_label = record[0][0]
        edge_label = record[1]
        dst_label = record[2][0]
        result.append((src_label, dst_label, edge_label))
    return result

def generate_id(key_value: object, label: str):
    '''
    Generates an id that is unique across all node labels
    '''
    node_id = f"{label}:{key_value}"
    __ID_VALUES__[node_id] = key_value
    return node_id

def id_to_value(node_id: str) -> object:
    '''
    Retrieves value from id
    '''
    return __ID_VALUES__[node_id]

def extract_nodes(tsv_file_path: str, node_keys: dict[str, str]):

    '''
    Extracts all nodes from graph and saves them into a tsv file
    '''

    with open(tsv_file_path, 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        for label, key in node_keys.items():
            # Query will return the keys of all nodes with given label
            records, _, _ = graph_utils.execute_query(f"MATCH (n:{label}) RETURN n.{key}")
            for record in records:
                node_id = generate_id(record[0], label)
                tsv_writer.writerow((node_id, label))

def extract_edges(tsv_file_path: str, node_keys: dict[str, str], relations: list[tuple[str, str, str]]):

    '''
    Extracts all edges from graph and saves them into a tsv file
    '''

    with open(tsv_file_path, 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        for src_label, dst_label, edge_label in relations:
            src_key = node_keys[src_label]
            dst_key = node_keys[dst_label]
            records, _, _ = graph_utils.execute_query(f"""
MATCH (n:{src_label})-[:{edge_label}]->(m:{dst_label})
RETURN n.{src_key}, m.{dst_key}""")
            for record in records:
                src_id = generate_id(record[0], src_label)
                dst_id = generate_id(record[1], dst_label)
                tsv_writer.writerow((src_id, dst_id, edge_label))

def extract_ontology(tsv_file_path: str):

    '''
    Creates an empty ontology tsv file
    '''

    with open(tsv_file_path, 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        tsv_writer.writerow(('Country', 'Region'))
        tsv_writer.writerow(('Industry', 'Sector'))
        # tsv_writer.writerow(('Industry', 'IndustrySector'))
        # tsv_writer.writerow(('Sector', 'IndustrySector'))
        pass

def fact_check(node_pairs: Iterable[tuple[str, str]],
               src_label: str,
               dst_label: str,
               edge_label: str,
               min_supp: float = 0.0001,
               min_conf: float = 0.0001,
               max_size: int = 2,
               top_k: int = 10,
               api_url: str = "http://localhost:8080/api/factchecker/check"):
    
    node_keys = extract_key_constraints()
    relations = extract_relations()

    with tempfile.TemporaryDirectory() as tmpdirname:

        graph_nodes_file = os.path.join(tmpdirname, "graph_nodes.tsv")
        graph_edges_file = os.path.join(tmpdirname, "graph_edges.tsv")
        graph_ontology_file = os.path.join(tmpdirname, "graph_ontology.tsv")
        input_edges_file = os.path.join(tmpdirname, "input_edges.tsv")

        extract_nodes(graph_nodes_file, node_keys)
        extract_edges(graph_edges_file, node_keys, relations)
        extract_ontology(graph_ontology_file)

        with open(input_edges_file, 'w') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            for src_key_value, dst_key_value in node_pairs:
                src_id = generate_id(src_key_value, src_label)
                dst_id = generate_id(dst_key_value, dst_label)
                tsv_writer.writerow((src_id, dst_id, edge_label))
        
        files = {
            'graphNodes': open(graph_nodes_file, 'rb'),
            'graphEdges': open(graph_edges_file, 'rb'),
            'graphOntology': open(graph_ontology_file, 'rb'),
            'inputEdges': open(input_edges_file, 'rb')
        }

        data = {
            'minSupp': min_supp,
            'minConf': min_conf,
            'maxSize': max_size,
            'topK': top_k
        }

        try:
            # Send the request
            response = requests.post(api_url, files=files, data=data)
            
            # Check if request was successful
            response.raise_for_status()

            result = response.json()
            
            return result
        
        except requests.exceptions.RequestException as e:
            print((f"API request failed: {str(e)}"))
            if hasattr(e.response, 'text'):
                print(f"Response content: {e.response.text}")
            raise
            
        finally:
            # Ensure files are closed even if an error occurs
            for file in files.values():
                file.close()


def fact_check_and_add(edges_to_add: tuple[str]):
    node_keys = extract_key_constraints()
    groups = defaultdict(list)
    all_patterns = defaultdict(list)
    for src_key_value, dst_key_value, src_label, dst_label, edge_label in edges_to_add:
        groups[(src_key_value, src_label, dst_label, edge_label)].append(dst_key_value)
    while groups:
        max_count = 0
        most_dups = set()
        for group, dst_key_values in groups.items():
            counts = Counter(dst_key_values)
            group_max_count = max(counts.values())
            if group_max_count > max_count:
                max_count = group_max_count
                most_dups = {group}
            elif group_max_count == max_count:
                most_dups.add(group)
        for group in most_dups:
            print(f"Processing group: {group}")
            src_key_value, src_label, dst_label, edge_label = group
            dst_key_values = set(groups[group])
            print(f"{len(dst_key_values)} edges found in group.")
            source_pairs = ((src_key_value, dst_key_value) for dst_key_value in dst_key_values)
            json_response = fact_check(source_pairs, src_label, dst_label, edge_label)
            n_patterns = len(json_response['patterns'])
            print(f"{n_patterns} patterns found for group.")

            filtered_edges = []

            for edge in json_response['results']:
                if n_patterns > 0 and edge['hits'] == 0:
                    continue
                filtered_edges.append({'src_key_value': id_to_value(edge['srcId']),
                                       'dst_key_value': id_to_value(edge['dstId'])})
            
            print(f"Adding {len(filtered_edges)} filtered edges.")

            src_key = node_keys[src_label]
            dst_key = node_keys[dst_label]

            query = f"""
MATCH
    (n:{src_label}{{{src_key}:$src_key_value}}),
    (m:{dst_label}{{{dst_key}:$dst_key_value}})
MERGE (n)-[:{edge_label}]->(m)"""
            
            graph_utils.execute_query_with_params(query, *filtered_edges)

            all_patterns[(src_label, dst_label, edge_label)].extend(json_response['patterns'])

            groups.pop(group)

            print()
    
    return all_patterns
