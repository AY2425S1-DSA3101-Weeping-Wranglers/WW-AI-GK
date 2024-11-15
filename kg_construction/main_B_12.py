'''
This script will perform KG construction and consistency checking.
Refer to `graph_construction.ipynb` for a walkthrough of the steps.
'''

import os
import re
import json
import requests
import pandas as pd
import urllib.request
from operator import itemgetter
import pyreadr
import numpy as np
from sentence_transformers import SentenceTransformer
import graph_utils
from fact_checking import fact_check_and_add, extract_all_patterns, visualize_rules


######################
# Neo4J AuraDB Setup #
######################
print("Resetting database...")
print("Removing all nodes and relationships")
graph_utils.reset_graph()
print("Removing all indexes and constraints")
graph_utils.reset_constraints()
print()

#########################################
# Adding Schema Constraints and Indexes #
#########################################
print("Adding constraints and indexes...")

print("Region Node")
graph_utils.execute_query('''
CREATE CONSTRAINT region_m49_key IF NOT EXISTS
FOR (r:Region) REQUIRE r.m49 IS NODE KEY''')
graph_utils.execute_query('''
CREATE CONSTRAINT region_name_unique IF NOT EXISTS
FOR (r:Region) REQUIRE r.name IS UNIQUE''')
graph_utils.execute_query('''
CREATE FULLTEXT INDEX region_name_index IF NOT EXISTS
FOR (r:Region) ON EACH [r.name]''')

print("Country Node")
graph_utils.execute_query('''
CREATE CONSTRAINT country_iso3_key IF NOT EXISTS
FOR (c:Country) REQUIRE c.iso3 IS NODE KEY''')
graph_utils.execute_query('''
CREATE CONSTRAINT country_iso2_unique IF NOT EXISTS
FOR (c:Country) REQUIRE c.iso2 IS UNIQUE''')
graph_utils.execute_query('''
CREATE CONSTRAINT country_name_unique IF NOT EXISTS
FOR (c:Country) REQUIRE c.name IS UNIQUE''')
graph_utils.execute_query('''
CREATE FULLTEXT INDEX country_aliases_index IF NOT EXISTS
FOR (c:Country) ON EACH [c.aliases]''')

print("Sector Node")
graph_utils.execute_query('''
CREATE CONSTRAINT sector_gics_key IF NOT EXISTS
FOR (s:Sector) REQUIRE s.gics IS NODE KEY''')
graph_utils.execute_query('''
CREATE CONSTRAINT country_name_unique IF NOT EXISTS
FOR (c:Country) REQUIRE c.name IS UNIQUE''')

print("Industry Node")
graph_utils.execute_query('''
CREATE CONSTRAINT industry_gics_key IF NOT EXISTS
FOR (i:Industry) REQUIRE i.gics IS NODE KEY''')
graph_utils.execute_query('''
CREATE CONSTRAINT industry_name_unique IF NOT EXISTS
FOR (i:Industry) REQUIRE i.name IS UNIQUE''')
graph_utils.execute_query('''
CREATE VECTOR INDEX industry_description_index IF NOT EXISTS
FOR (i:Industry)
ON i.embedding
OPTIONS { indexConfig: {
 `vector.quantization.enabled`: false
}}''')

print("Company Node")
graph_utils.execute_query('''CREATE CONSTRAINT company_ticker_key IF NOT EXISTS
FOR (c:Company) REQUIRE c.ticker IS NODE KEY''')
graph_utils.execute_query('''CREATE FULLTEXT INDEX company_names_index IF NOT EXISTS
FOR (c:Company) ON EACH [c.names]''')

print()

#######################
# Adding Initial Data #
#######################
print("Adding Initial Data...")

print("Region Nodes")
df_m49 = pd.read_csv('../data/UNSD_m49.csv', sep=';')
continents = df_m49[['Region Code', 'Region Name']]\
                    .dropna()\
                    .drop_duplicates()\
                    .rename(columns={
                        'Region Code': 'm49',
                        'Region Name': 'name'
                    })
subregions = df_m49[['Sub-region Code', 'Sub-region Name']]\
                    .dropna()\
                    .drop_duplicates()\
                    .rename(columns={
                        'Sub-region Code': 'm49',
                        'Sub-region Name': 'name'
                    })
itdregions = df_m49[['Intermediate Region Code', 'Intermediate Region Name']]\
                    .dropna()\
                    .drop_duplicates()\
                    .rename(columns={
                        'Intermediate Region Code': 'm49',
                        'Intermediate Region Name': 'name'
                    })
regions = pd.concat([continents, subregions, itdregions], ignore_index=True)\
            .astype({'m49': int})
region_nodes = regions.to_dict('records')
graph_utils.execute_query_with_params("MERGE (:Region{m49: $m49, name: $name})",
                                      *region_nodes)

print("Country Nodes")
countries = df_m49[['ISO-alpha3 Code', 'ISO-alpha2 Code', 'Country or Area']]\
                    .dropna()\
                    .drop_duplicates()\
                    .rename(columns={
                        'ISO-alpha3 Code': 'iso3',
                        'ISO-alpha2 Code': 'iso2',
                        'Country or Area': 'name'
                    })
country_nodes = countries.to_dict('records')
graph_utils.execute_query_with_params("MERGE (:Country{iso3: $iso3, name: $name, iso2: $iso2})",
                                      *country_nodes)

print("Country IS_IN Region Relationships")
country_continent = df_m49[['ISO-alpha3 Code', 'Region Code']]\
                            .dropna()\
                            .drop_duplicates()\
                            .rename(columns={
                                'ISO-alpha3 Code': 'iso3',
                                'Region Code': 'm49'
                            })
country_subregion = df_m49[['ISO-alpha3 Code', 'Sub-region Code']]\
                            .dropna()\
                            .drop_duplicates()\
                            .rename(columns={
                                'ISO-alpha3 Code': 'iso3',
                                'Sub-region Code': 'm49'
                            })
country_itdregion = df_m49[['ISO-alpha3 Code', 'Intermediate Region Code']]\
                            .dropna()\
                            .drop_duplicates()\
                            .rename(columns={
                                'ISO-alpha3 Code': 'iso3',
                                'Intermediate Region Code': 'm49'
                            })
country_region = pd.concat([country_continent, country_subregion, country_itdregion], ignore_index=True)
isin_relationships = country_region.to_dict('records')
graph_utils.execute_query_with_params('''
MATCH
    (c:Country{iso3: $iso3}),
    (r:Region{m49: $m49})
MERGE (c)-[:IS_IN]->(r)''', *isin_relationships)

print("Country Aliases")
def split_alias(row):
    '''
    Splits a row if Alias contains multiple aliases seperated by " or "
    '''
    if ' or ' in row['Alias']:
        values = row['Alias'].split(' or ')
        return pd.DataFrame({'iso3': [row['iso3']] * len(values), 'Alias': values})
    return pd.DataFrame({'iso3': [row['iso3']], 'Alias': [row['Alias']]})
df_alias = pd.read_csv('../data/country_aliases.csv')
aliases = pd.concat([split_alias(row) for _, row in df_alias.iterrows()],
                  ignore_index=True)\
        .dropna()\
        .drop_duplicates()\
        .rename(columns={'Alias': 'alias'})
country_aliases = aliases.to_dict('records')
graph_utils.execute_query_with_params('''
MERGE (c:Country {iso3: $iso3})
SET c.aliases = 
    CASE
        WHEN c.aliases IS NULL THEN [$alias]
        WHEN NOT $alias IN c.aliases THEN c.aliases + $alias
        ELSE c.aliases
    END''', *country_aliases)

print("Country Statistics")
def get_worldbank(indicator: str) -> pd.DataFrame:
    '''
    Get indicator data using worldbank API
    '''
    with urllib.request.urlopen(f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000") as url:
        data = json.load(url)[1]
    ind = data[0]['indicator']['value']
    iso3 = map(itemgetter('countryiso3code'), data)
    year = map(itemgetter('date'), data)
    value = map(itemgetter('value'), data)
    return pd.DataFrame({
        'iso3': iso3,
        'year': year,
        ind: value
    }).replace('', np.nan)\
      .dropna()\
      .set_index(['iso3', 'year'])
population = get_worldbank('SP.POP.TOTL')
gdp = get_worldbank('NY.GDP.MKTP.CD')
pv = get_worldbank('PV.EST')
ctr = pd.read_excel('../data/corp_tax_rate.xlsx')\
        .melt(id_vars='iso_3',
              value_vars=range(1980, 2024),
              var_name='year',
              value_name='corporate_tax_rate')\
        .rename(columns={'iso_3': 'iso3'})\
        .astype({'year': str})\
        .set_index(['iso3', 'year'])
stats = pd.concat([population, gdp, pv, ctr], axis=1).sort_index()\
          .reset_index()\
          .rename(columns={
              'Population, total': 'population',
              'GDP (current US$)': 'gdp',
              'Political Stability and Absence of Violence/Terrorism: Estimate': 'pv',
              'corporate_tax_rate': 'corporate_tax_rate'
          })
country_stats = stats[stats['year'] == '2022'].to_dict('records')
graph_utils.execute_query_with_params('''
MATCH (c:Country {iso3: $iso3})
SET
    c.population = $population,
    c.gdp = $gdp,
    c.pv = $pv,
    c.corporate_tax_rate = $corporate_tax_rate''', *country_stats)

print("Sector Nodes")
gics_url = 'https://github.com/bautheac/GICS/raw/0c2b0e4c0ca56a0e520301fd978fc095ed4fc328/data/standards.rda'
gics_response = requests.get(gics_url)
rda_file_path = '../data/standards.rda'
with open(rda_file_path, 'wb') as file:
    file.write(gics_response.content)
result = pyreadr.read_r(rda_file_path)
gics = result[list(result.keys())[0]]
os.remove(rda_file_path)
def gics_wrangling(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.rename(columns={
        'sector id': 'sector_id',
        'sector name': 'sector_name',
        'industry group id': 'industry_group_id',
        'industry group name': 'industry_group_name',
        'industry id': 'industry_id',
        'industry name': 'industry_name',
        'subindustry id': 'subindustry_id',
        'subindustry name': 'subindustry_name',
        'description': 'primary_activity'
    })
    df['sector_id'] = df['sector_id'].astype('Int64')  
    df['industry_group_id'] = df['industry_group_id'].astype('Int64')
    df['industry_id'] = df['industry_id'].astype('Int64')
    df['subindustry_id'] = df['subindustry_id'].astype('Int64')
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    return df
df_standards = gics_wrangling(gics)
sector = df_standards[['sector_id', 'sector_name']] \
        .drop_duplicates() \
        .rename(columns={
            'sector_id': 'gics',
            'sector_name': 'name'
        })
sector_nodes = sector.to_dict('records')
graph_utils.execute_query_with_params("MERGE (:Sector{gics: $gics, name: $name})", *sector_nodes)

print("Industry Nodes")
industry = df_standards[['subindustry_id', 'subindustry_name', 'primary_activity']] \
           .drop_duplicates() \
           .rename(columns={
               'subindustry_id': 'gics',
               'subindustry_name': 'name',
               'primary_activity': 'description'
            })
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
industry_desc_embed = EMBEDDING_MODEL.encode(industry['description'].to_numpy())
industry['embedding'] = list(map(list, industry_desc_embed))
insustry_nodes = industry.to_dict('records')
graph_utils.execute_query_with_params("MERGE (:Industry{gics: $gics, name: $name, description: $description, embedding: $embedding})", *insustry_nodes)

print("Industry PART_OF Sector Relationships")
industry_sector = df_standards[['subindustry_id', 'sector_id']] \
                  .drop_duplicates() \
                  .rename(columns={
                      'subindustry_id': 'industry_gics',
                      'sector_id': 'sector_gics'
                  })
part_of_relationships = industry_sector.to_dict('records')
graph_utils.execute_query_with_params('''
MATCH
    (i:Industry{gics: $industry_gics}),
    (s:Sector{gics: $sector_gics})
MERGE (i)-[:PART_OF]->(s)''', *part_of_relationships)
print()

#################################
# Adding Extracted Company Data #
#################################
print("Processing Extracted Company Data...")
extracted_file = os.getenv("ER_EXTRACTION_OUTPUT", "merged_output.json")
with open(f'../output/{extracted_file}', 'r') as file:
    merged_json = json.load(file)

def is_valid_ticker(ticker_code):
    """Helper function to check if the ticker code is valid (str, 4 to 5 letters, all upper case)."""
    return isinstance(ticker_code, str) and 4 <= len(ticker_code) <= 5 and ticker_code.isupper()

def remove_invalid_ticker_companies(data):
    """Remove companies whose ticker_code doesn't meet the 3, 4, or 5 letter criteria."""
    if isinstance(data, str):
        print("Warning: data is a string, attempting to load as JSON.")
        data = json.loads(data)  
    
    if "nodes" in data and "Company" in data["nodes"]:
        companies = data["nodes"]["Company"]
        filtered_companies = [company for company in companies if is_valid_ticker(company.get("ticker_code"))]
        data["nodes"]["Company"] = filtered_companies
    else:
        print("Warning: The expected structure is not found in the data.")

    return data

def is_more_comprehensive(entry1, entry2):
    """Helper function to determine which duplicate has more comprehensive details."""
    return sum(1 for v in entry1.values() if v) > sum(1 for v in entry2.values() if v)

def clean_names(name):
    name = name.title()
    name = re.sub(r'[^\w\s]', '', name)
    return name

def standarize_case(data):
    """function to standardize title case and no other special."""
    for company in data["nodes"]["Company"]:
        company["name"] = clean_names(company["name"])
    return data

def company_validate(data): 
    data = remove_invalid_ticker_companies(data)
    data = standarize_case(data)
    print("Completed.")
    return data

validated_data = company_validate(merged_json)

print("Adding Company Nodes")
companies = validated_data['nodes']['Company']
for company in companies:
    company['founded_year'] = company['founded_year'] or ""
graph_utils.execute_query_with_params('''
MERGE (c:Company {ticker: $ticker_code})
SET c.names = 
    CASE
        WHEN c.names IS NULL THEN [$name]
        WHEN NOT $name IN c.names THEN c.names + $name
        ELSE c.names
    END,
    c.founded_year = $founded_year''', *companies)

print()

print("Preparing Company relationships")
all_edges = []

print("Company-Industry relationships")
is_involved_in_data = validated_data['relationships']['IS_INVOLVED_IN']
for company_industry in is_involved_in_data:
    industry_name = company_industry['industry_name']
    company_industry['company_name'] = clean_names(company_industry['company_name'])
    company_industry['embedding'] = EMBEDDING_MODEL.encode(industry_name).tolist()
produces_data = validated_data['relationships']['PRODUCES']
for company_product in produces_data:
    product_name = company_product['product_name']
    company_product['company_name'] = clean_names(company_product['company_name'])
    company_product['embedding'] = EMBEDDING_MODEL.encode(product_name).tolist()
is_involved_in_edges = graph_utils.execute_query_with_params("""
CALL db.index.fulltext.queryNodes('company_names_index', $company_name)
    YIELD node AS c, score AS company_score
CALL db.index.vector.queryNodes('industry_description_index', 10, $embedding)
    YIELD node AS i, score AS industry_score
WHERE company_score > 1
AND industry_score > 0.7
RETURN
    c.ticker AS ticker,
    i.gics AS gics""", *(is_involved_in_data + produces_data))
for records, _, _ in is_involved_in_edges:
    for ticker, gics in records:
        all_edges.append((ticker, gics, "Company", "Industry", "IS_INVOLVED_IN", {}))

print("Company-Country relationships")
headquarters_data = validated_data['relationships']['HEADQUARTERS_IN']
for entry in headquarters_data:
    entry['company_name'] = clean_names(entry['company_name'])
headquarters_edges = graph_utils.execute_query_with_params("""
CALL db.index.fulltext.queryNodes('company_names_index', $company_name)
    YIELD node AS company, score AS company_score
CALL db.index.fulltext.queryNodes('country_aliases_index', $country_name)
    YIELD node AS country, score AS country_score
WHERE company_score > 1
AND country_score > 1
RETURN company.ticker AS ticker, country.iso3 AS iso3""", *headquarters_data)
for records, _, _ in headquarters_edges:
    for ticker, iso3 in records:
        all_edges.append((ticker, iso3, "Company", "Country", "HEADQUARTERS_IN", {}))

operates_data = validated_data['relationships']['OPERATES_IN_COUNTRY']
for entry in operates_data:
    entry['company_name'] = clean_names(entry['company_name'])
    entry['net_sales'] = entry.pop('net sales')
operates_edges = graph_utils.execute_query_with_params("""
CALL db.index.fulltext.queryNodes('company_names_index', $company_name)
    YIELD node AS company, score AS company_score
CALL db.index.fulltext.queryNodes('country_aliases_index', $country_name)
    YIELD node AS country, score AS country_score
WHERE company_score > 1
AND country_score > 1
RETURN
    company.ticker AS ticker,
    country.iso3 AS iso3,
    $headcount AS headcount,
    $net_sales AS net_sales""", *operates_data)
for records, _, _ in operates_edges:
    for ticker, iso3, headcount, net_sales in records:
        all_edges.append((ticker, iso3, "Company", "Country", "OPERATES_IN",
                          {'headcount': headcount, 'net_sales': net_sales}))

print("Company-Company relationships")
company_competes = validated_data['relationships']['COMPETES_WITH']
for entry in company_competes:
    entry['company_name_1'] = clean_names(entry['company_name_1'])
    entry['company_name_2'] = clean_names(entry['company_name_2'])
competes_edges = graph_utils.execute_query_with_params("""
CALL db.index.fulltext.queryNodes('company_names_index', $company_name_1)
    YIELD node AS company1, score AS c1_score
CALL db.index.fulltext.queryNodes('company_names_index', $company_name_2)
    YIELD node AS company2, score AS c2_score
WHERE c1_score > 1
AND c2_score > 1
RETURN company1.ticker AS ticker1, company2.ticker AS ticker2""", *company_competes)
for records, _, _ in competes_edges:
    for ticker1, ticker2 in records:
        all_edges.append((ticker1, ticker2, "Company", "Company", "COMPETES_WITH", {}))

company_subsidiary = validated_data['relationships']['SUBSIDIARY_OF']
for entry in company_subsidiary:
    entry['company_name_1'] = clean_names(entry['company_name_1'])
    entry['company_name_2'] = clean_names(entry['company_name_2'])
subsidiary_edges = graph_utils.execute_query_with_params("""
CALL db.index.fulltext.queryNodes('company_names_index', $company_name_1)
    YIELD node AS company1, score AS c1_score
CALL db.index.fulltext.queryNodes('company_names_index', $company_name_2)
    YIELD node AS company2, score AS c2_score
WHERE c1_score > 1
AND c2_score > 1
RETURN company1.ticker AS ticker1, company2.ticker AS ticker2""", *company_subsidiary)
for records, _, _ in subsidiary_edges:
    for ticker1, ticker2 in records:
        all_edges.append((ticker1, ticker2, "Company", "Company", "SUBSIDIARY_OF", {}))

company_supplies = validated_data['relationships']['PARTNERS_WITH']
for entry in company_supplies:
    entry['company_name_1'] = clean_names(entry['company_name_1'])
    entry['company_name_2'] = clean_names(entry['company_name_2'])
subsidiary_edges = graph_utils.execute_query_with_params("""
CALL db.index.fulltext.queryNodes('company_names_index', $company_name_1)
    YIELD node AS company1, score AS c1_score
CALL db.index.fulltext.queryNodes('company_names_index', $company_name_2)
    YIELD node AS company2, score AS c2_score
WHERE c1_score > 1
AND c2_score > 1
RETURN company1.ticker AS ticker1, company2.ticker AS ticker2""", *company_supplies)
for records, _, _ in subsidiary_edges:
    for ticker1, ticker2 in records:
        all_edges.append((ticker1, ticker2, "Company", "Company", "SUPPLIES_TO", {}))
print()

########################
# Consistency Checking #
########################
print("Consistency Checking and Adding Company Relationships...")
fact_check_and_add(all_edges, min_supp=0.5, min_conf=0.1, top_k=50, max_size=2)

print("Extracting Patterns from Final Graph...")
patterns = extract_all_patterns(min_supp=0.5, min_conf=0.1, top_k=50, max_size=2)
visualize_rules(patterns)