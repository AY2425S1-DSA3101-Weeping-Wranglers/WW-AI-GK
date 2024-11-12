import os
import csv
import requests
import tempfile
from collections import defaultdict
from collections.abc import Iterable
import networkx as nx
import matplotlib.pyplot as plt
import graph_utils

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


def extract_relations() -> set[tuple[str, str, str]]:
    '''
    Extracts all relation types from graph
    '''
    records, _, _ = graph_utils.execute_query("""
MATCH (n)-[r]->(m)
RETURN DISTINCT
    LABELS(n),
    TYPE(r),
    LABELS(m)""")
    result = set()
    for record in records:
        src_label = record[0][0]
        edge_label = record[1]
        dst_label = record[2][0]
        result.add((src_label, dst_label, edge_label))
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


def extract_nodes(node_keys: dict[str, str]) -> Iterable[tuple[str, str]]:
    '''
    Extracts all nodes from graph
    '''
    for label, key in node_keys.items():
        # Query will return the keys of all nodes with given label
        records, _, _ = graph_utils.execute_query(f"MATCH (n:{label}) RETURN n.{key}")
        for record in records:
            node_id = generate_id(record[0], label)
            yield (node_id, label)


def extract_edges(node_keys: dict[str, str],
                  relations: set[tuple[str, str, str]]) -> Iterable[tuple[str, str, str]]:
    '''
    Extracts all edges from graph
    '''
    for src_label, dst_label, edge_label in relations:
        src_key = node_keys[src_label]
        dst_key = node_keys[dst_label]
        records, _, _ = graph_utils.execute_query(f"""
MATCH (n:{src_label})-[:{edge_label}]->(m:{dst_label})
RETURN n.{src_key}, m.{dst_key}""")
        for record in records:
            src_id = generate_id(record[0], src_label)
            dst_id = generate_id(record[1], dst_label)
            yield (src_id, dst_id, edge_label)


def extract_ontology(node_keys: dict[str, str]) -> Iterable[tuple[str, str]]:
    '''
    Creates an empty ontology tsv file
    '''
    for label in node_keys:
        yield (label, 'Entity')


def write_tsv(tsv_file: str, rows: Iterable[tuple], append: bool=False):
    '''
    Writes an interable of rows into a tsv file
    '''
    mode = 'a' if append else 'w'
    with open(tsv_file, mode) as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        for row in rows:
            tsv_writer.writerow(row)


def fact_check(graph_nodes_file: str,
               graph_edges_file: str,
               graph_ontology_file: str,
               input_edges_file: str,
               min_supp: float = 0.5,
               min_conf: float = 0.1,
               max_size: int = 2,
               top_k: int = 50,
               api_url: str = "http://localhost:8080/api/factchecker/check") -> dict:
    '''
Calls FactChecker API. Given an input graph and input edges (of the same relation type), perform the following:
1. Generate patterns (GFCs) for the given relation and input graph.
2. Checks each input edge against each found pattern.

The pattern mining relies on Principal Closed World Assumption (PCWA) of the input graph,
i.e. if the graph has at least one edge (V1)-[:R]->(V2),
we assume that we have complete information of all (V1)-[:R]->(Vx).
For example if the graph has two competitors of Apple,
    (Apple)-[:COMPETES_WITH]->(Google) and
    (Apple)-[:Competes_With]->(Samsung),
anything not in the graph (e.g. (Apple)-[:COMPETES_WITH]->(Meta)) is considered false.
Read more about GFCs in our [repository](https://github.com/001waiyan/GDRB),
which was forked from the [original paper](https://github.com/001waiyan/GDRB/blob/master/2018-DASFAA-GFC-paper.pdf)'s repository
for the purposes of this project.

Parameters
- graph_nodes_file: Path to TSV file containing graph nodes
- graph_edges_file: Path to TSV file containing graph edges
- graph_ontology_file: Path to TSV file containing graph ontology
- input_edges_file: Path to TSV file containing edges to test
- min_supp: Minimum support of GFCs (Range: 0.0 to 1.0)
- min_conf: Minimum confidence of GFCs (Range: 0.0 to 1.0)
- max_size: Maximum size of extracted patterns
- top_k: Number of patterns extracted for each relation
- api_url: URL of the FactChecker API

Returns a dictionary with the format:
"patterns": list of objects, topK patterns
⎿ relations: list of objects, relations in the extracted pattern
    ⎿  srcLabel: string, label of source node in pattern relation
    ⎿  dstLabel: string, label of destination node in pattern relation
    ⎿  edgeLabel: string, label of pattern relation
⎿  supp: double, support of pattern
⎿  conf: double, confidence of pattern
"results": list of objects, fact checking scores of input edges
⎿  srcId: string, id of source node in edge
⎿  dstId: string, id of destination node in edge
⎿  edgeLabel: string, label of edge
⎿  hits: int, number of patterns that cover the edge
    '''
        
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


def add_group_edges(node_keys: dict[str, str],
                    src_key_value: object,
                    src_label: str,
                    dst_label: str,
                    edge_label: str,
                    dst_properties: dict[str, dict[str, object]]):
    '''
    Adds edges of the same group to the Neo4j database.
    A group has edges of the same source node and relation type.
    '''

    group_edges = []
    property_names = tuple(next(iter(dst_properties.values())).keys())
    for dst_key_value, properties in dst_properties.items():
        entry = {}
        entry["src_key_value"] = src_key_value
        entry["dst_key_value"] = dst_key_value
        for property_name in property_names:
            entry[property_name] = properties[property_name]
        group_edges.append(entry)
    src_key = node_keys[src_label]
    dst_key = node_keys[dst_label]
    if property_names:
        query = f"""
MATCH
    (n:{src_label}{{{src_key}:$src_key_value}}),
    (m:{dst_label}{{{dst_key}:$dst_key_value}})
MERGE (n)-[r:{edge_label}]->(m)
SET """ + ", ".join(f"r.{property_name} = ${property_name}"
                    for property_name in property_names)
    else:
        query = f"""
MATCH
    (n:{src_label}{{{src_key}:$src_key_value}}),
    (m:{dst_label}{{{dst_key}:$dst_key_value}})
MERGE (n)-[:{edge_label}]->(m)"""
    graph_utils.execute_query_with_params(query, *group_edges)


def fact_check_and_add(edges_to_add: tuple,
                       min_supp: float = 0.0001,
                       min_conf: float = 0.0001,
                       max_size: int = 2,
                       top_k: int = 10,
                       api_url: str = "http://localhost:8080/api/factchecker/check"):
    
    '''
Systematically fact checks and adds to the Neo4j database a list of edges, while maintaining PCWA of the graph. The edges to add are grouped by same source node and relation type (e.g. all COMPETES_WITH edges for the Apple node). Each group is fact checked using the FactChecker API and added one at a time to maintain PCWA.

Algorithm used:
1. Group the edges by the same source node and relation type.
2. Add all groups that contain duplicate edges to the graph, without fact checking. (Assuming that these edges are more likely to be consistent)
3. For the remaining groups, run the FactChecker API on each of them.
4. If an edge was matched by at least one found pattern, add that edge to the graph.
5. If no patterns were found for the group, add all its edges to the graph.

FactChecker API: Given an input graph and input edges (of the same relation type), perform the following:
1. Generate patterns (GFCs) for the given relation and input graph.
2. Checks each input edge against each found pattern.

The pattern mining relies on Principal Closed World Assumption (PCWA) of the input graph, i.e. if the graph has at least one edge (V1)-[:R]->(V2), we assume that we have complete information of all (V1)-[:R]->(Vx). For example if the graph has two competitors of Apple, (Apple)-[:COMPETES_WITH]->(Google) and (Apple)-[:Competes_With]->(Samsung), anything not in the graph (e.g. (Apple)-[:COMPETES_WITH]->(Meta)) is considered false. Read more about GFCs in our [repository](https://github.com/001waiyan/GDRB), which was forked from the [original paper](https://github.com/001waiyan/GDRB/blob/master/2018-DASFAA-GFC-paper.pdf)'s repository for the purposes of this project.
    '''
    
    node_keys = extract_key_constraints()
    groups = defaultdict(list)
    for src_key_value, dst_key_value, src_label, dst_label, edge_label, properties in edges_to_add:
        groups[(src_key_value, src_label, dst_label, edge_label)].append((dst_key_value, properties))

    groups_without_dups = {}

    print("="*20 + "Adding groups with duplicates" + "="*20)
    for group, dst_properties in groups.items():
        uniq_dst_properties = dict(dst_properties)
        if len(uniq_dst_properties) == len(dst_properties):
            # no dups, add later
            groups_without_dups[group] = dst_properties
            continue
        print(f"Processing group: {group}")
        print(f"{len(dst_properties)} edges found in group.")
        print(f"Adding {len(uniq_dst_properties)} unique edges.")
        src_key_value, src_label, dst_label, edge_label = group
        add_group_edges(node_keys, src_key_value, src_label, dst_label, edge_label, uniq_dst_properties)
        print()

    all_patterns = defaultdict(set)
    with tempfile.TemporaryDirectory() as tmpdirname:
        relations = extract_relations()
        graph_nodes = extract_nodes(node_keys)
        graph_edges = extract_edges(node_keys, relations)
        ontology = extract_ontology(node_keys)

        graph_nodes_file = os.path.join(tmpdirname, "graph_nodes.tsv")
        graph_edges_file = os.path.join(tmpdirname, "graph_edges.tsv")
        graph_ontology_file = os.path.join(tmpdirname, "graph_ontology.tsv")
        input_edges_file = os.path.join(tmpdirname, "input_edges.tsv")

        write_tsv(graph_nodes_file, graph_nodes)
        write_tsv(graph_edges_file, graph_edges)
        write_tsv(graph_ontology_file, ontology)
        
        print("="*20 + "Adding groups without duplicates" + "="*20)
        for group in groups_without_dups:
            print(f"Processing group: {group}")
            src_key_value, src_label, dst_label, edge_label = group
            dst_properties = dict(groups_without_dups[group])
            print(f"{len(dst_properties)} edges found in group.")
            input_edges = []
            src_id = generate_id(src_key_value, src_label)
            for dst_key_value in dst_properties:
                dst_id = generate_id(dst_key_value, dst_label)
                input_edges.append((src_id, dst_id, edge_label))
            write_tsv(input_edges_file, input_edges)
            json_response = fact_check(graph_nodes_file,
                                       graph_edges_file,
                                       graph_ontology_file,
                                       input_edges_file,
                                       min_supp, min_conf, max_size, top_k, api_url)
            patterns = json_response['patterns']
            print(f"{len(patterns)} patterns found for group.")

            filtered_edges = []
            for edge in json_response['results']:
                dst_id = edge['dstId']
                if patterns and edge['hits'] == 0:
                    dst_key_value = id_to_value(dst_id)
                    dst_properties.pop(dst_key_value)
                else:
                    filtered_edges.append((src_id, dst_id, edge_label))
            
            print(f"Adding {len(dst_properties)} filtered edges.")
            
            if dst_properties:
                add_group_edges(node_keys, src_key_value, src_label, dst_label, edge_label, dst_properties)

            # update tsv file with added edges
            write_tsv(graph_edges_file, filtered_edges, append=True)

            all_patterns[(src_label, dst_label, edge_label)].update(map(Pattern, patterns))
            print()
    
    return all_patterns

class Pattern:
    def __init__(self, json_object: dict):
        self.supp = json_object['supp']
        self.conf = json_object['conf']
        self.node_labels = tuple(json_object['nodes'])
        self.src_label = self.node_labels[0]
        self.dst_label = self.node_labels [1]
        self.node_names = tuple(f"{label}{i}" for i, label in enumerate(self.node_labels, 1))
        edges = []
        for edge in json_object['edges']:
            for edge_label in edge['edgeLabel'].split('&'):
                edges.append((edge['srcId'], edge['dstId'], edge_label))
        edges.sort()
        self.edges = tuple(edges)
    
    def __eq__(self, value: object) -> bool:
        return isinstance(value, Pattern) and (self.node_labels, self.edges) == (value.node_labels, value.edges)

    def __hash__(self) -> int:
        return hash((self.node_labels, self.edges))
    
    def __repr__(self) -> str:
        return "Pattern(" + ",\n".join(f"({self.node_names[u]})-[:{edge_label}]->({self.node_names[v]})"
                                      for u, v, edge_label in self.edges) + ")"


def visualize_rule(pattern: Pattern, rule_head: tuple[str, str, str]):
    """
    Visualize a single rule with its pattern (body) and head.
    
    Args:
        pattern: Pattern object containing the rule body
        rule_head: tuple of (src_label, dst_label, edge_label) representing the rule head
    """
    G = nx.DiGraph()
    
    for i, label in enumerate(pattern.node_labels):
        G.add_node(i, label=label)
    
    # Add edges from the pattern
    for src, dst, label in pattern.edges:
        G.add_edge(src, dst, label=label, is_rule_head=False)
    
    # Add rule head edge (between anchored nodes)
    G.add_edge(0, 1, label=rule_head[2], is_rule_head=True)
    
    plt.figure(figsize=(12, 8))
    
    pos = nx.spring_layout(G, k=1.5)
    
    # Calculate node size based on label length
    node_sizes = [2000 for _ in range(len(pattern.node_labels))]
    
    # Draw nodes
    node_colors = ['lightgreen' if i < 2 else 'lightblue' for i in range(len(pattern.node_labels))]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    
    # Function to shorten edges to prevent overlap with nodes
    def shorten_edge(pos, source, target, scale=0.8):
        """Returns the position to shorten an edge so it doesn't overlap with nodes."""
        source_pos = pos[source]
        target_pos = pos[target]
        
        vx = target_pos[0] - source_pos[0]
        vy = target_pos[1] - source_pos[1]
        
        norm = (vx**2 + vy**2)**0.5
        
        if norm != 0:
            vx = vx / norm
            vy = vy / norm
        
        start_x = source_pos[0] + vx * (1 - scale)
        start_y = source_pos[1] + vy * (1 - scale)
        end_x = target_pos[0] - vx * (1 - scale)
        end_y = target_pos[1] - vy * (1 - scale)
        
        return (start_x, start_y), (end_x, end_y)

    # Draw edges with shortened paths
    for (u, v, d) in G.edges(data=True):
        is_rule_head = d['is_rule_head']
        start_pos, end_pos = shorten_edge(pos, u, v)
        
        # Create edge path
        edge_path = plt.matplotlib.patches.FancyArrowPatch(
            start_pos, end_pos,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0',
            mutation_scale=20,
            linestyle='dashed' if is_rule_head else 'solid',
            color='red' if is_rule_head else 'black',
            shrinkA=0,
            shrinkB=0
        )
        plt.gca().add_patch(edge_path)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, 
                          {i: f"{label}" if i < 2 else label 
                           for i, label in enumerate(pattern.node_labels)})
    
    # Draw edge labels with different colors for pattern and rule head
    edge_labels = {}
    for (u, v, d) in G.edges(data=True):
        start_pos, end_pos = shorten_edge(pos, u, v, scale=0.5)
        edge_labels[(u, v)] = {
            'label': d['label'],
            'pos': ((start_pos[0] + end_pos[0])/2, (start_pos[1] + end_pos[1])/2),
            'color': 'red' if d['is_rule_head'] else 'black'
        }
    
    # Draw edge labels
    for (u, v), label_info in edge_labels.items():
        plt.text(label_info['pos'][0], label_info['pos'][1], 
                label_info['label'],
                color=label_info['color'],
                horizontalalignment='center',
                verticalalignment='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.title(f"Rule: {pattern.node_labels[0]} - {rule_head[2]} → {pattern.node_labels[1]}")
    
    plt.axis('off')
    return plt


def visualize_rules(rules_dict: dict):
    """
    Visualize all rules in the dictionary. Saves images in ./rules
    
    Args:
        rules_dict: Dictionary mapping (src_label, dst_label, edge_label) to list of Patterns
    """
    if not os.path.exists('rules'):
        os.makedirs('rules')
    for rule_head, patterns in rules_dict.items():
        for i, pattern in enumerate(patterns):
            plt = visualize_rule(pattern, rule_head)
            plt.savefig(f"rules/{rule_head[0]}_{rule_head[1]}_{rule_head[2]}_{i}.png")
            plt.close()
