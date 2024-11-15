# WW-AI-GK

This is the Weeping Wranglers' AI-Generated Knowledge Graph for Enterprise Content Management (ECM) Systems.

The NASDAQ exchange, which tracks over 3,000 stocks with a focus on tech and high-growth sectors, presents a complex challenge for investors performing due diligence. The current process of scrutinizing financial statements, news articles, and public data sources demands significant time and resources from investors and venture capital firms. Our knowledge graph solution addresses this challenge by integrating these diverse data sources into a unified, queryable format that enables rapid analysis and decision-making.

Through generating the knowledge graph, we hope to support investors by providing a comprehensive view of a company's ecosystem, partners, and competitors, which can serve as a holistic basis to assess a company’s fundamental value, sphere of influence, and growth potential.

## Datasets

- **SEC 10-K Filings**: https://sec-api.io/resources/extract-textual-data-from-edgar-10-k-filings-using-python
  URLs of the top 40 NASDAQ companies were written to NASDAQ_10-K_urls.xlsx through referencing their CIK numbers, which were then used to extract 10-K filings using the SEC-API. The Form 1 and Form 7 sections were stored in a mySQL database and exported to sqlite3 file for persistent storage.
- **Wikipedia Articles** Using the yahooquery library, the official websites of companies are retrieved and used to find their related Wikipedia pages, extracting relevant information using Beautiful Soup's HTML parser.
- **M49 Codes**: https://unstats.un.org/unsd/methodology/m49/overview/
  Standardised region and country codes by The United Nations Statistics Division. These are used to augment our knowledge graph with country and region nodes, and the IS_IN relationships between them.
- **Country Aliases**: https://www.kaggle.com/datasets/wbdill/country-aliaseslist-of-alternative-country-names
  Dataset of variations of country names, used for entity disambiguation for country nodes.
- **Corporate Tax Rates**: https://taxfoundation.org/data/all/global/corporate-tax-rates-by-country-2023/
  Annual corporate tax rates of countries around the world, collated by The Tax Foundation.
- **World Bank Open Data**: https://data.worldbank.org/
  Global development data used to add statistics, such as population, GDP, and political stability, to country nodes.
- **Global Industry Classification Standards (GICS)**: https://github.com/bautheac/GICS
  A standardised hierarchical industry classification system used to augment our knowledge graph with industry nodes.


## Subtask 1: Entity and Relationship Extraction
The **Diffbot Natural Language API** serves as the cornerstone of our entity extraction process. Released in September 2020, this API specializes in automatically building knowledge graphs from text through sophisticated natural language processing techniques. The API's capabilities are particularly powerful for our use case because it distinguishes between 69 different entity types, allowing us to capture the full complexity of corporate structures and relationships. Each entity receives a salience score that measures its relevance to the analyzed text, helping us prioritize the most significant information. These are then loaded into a JSON schema to organise them in a structured format. 

Entity resolution can be performed during this stage via **the ReFiNED Transformer**, which links entity mentions in documents to their corresponding Wikidata entries from a database of over 30 million entities, or during knowledge graph construction by using **vector indexes** to disambiguate entities whose vectors have high pairwise similarity scores. 

To scale up our ER Extraction Pipeline, we can build a distributed processing architecture that begins with **data ingestion into distributed storage systems** through the Hadoop Distributed File System or Amazon S3. These systems provide redundancy and fault tolerance for our large-scale unstructured data. We process this data using the **Spark DataFrame API**, which distributes aggregation tasks across multiple worker nodes for parallel processing. 
We **batch our API requests** to minimize network overhead and maintain a **caching system** for frequently accessed nodes and queries. Our caching system can maintain frequently accessed data in memory using a Least Recently Used policy, which significantly reduces database load and improves response times.
To achieve effective resource management, we **integrate our Spark API with YARN**, which dynamically allocates resources based on current workload demands. Data can also be **partitioned**, segmenting information by company, region, and industry to ensure balanced workload distribution. 


## Subtask 2: Knowledge Graph Construction and Verification

<<<<<<< Updated upstream
=======
## Project Setup

### 1. Repository Setup

1. Git clone this repository, initialising and updating all the submodules.

```bash
git clone --recurse-submodules https://github.com/AY2425S1-DSA3101-Weeping-Wranglers/WW-AI-GK
```

  Alternatively if you already cloned the project and did not include`--recurse-submodules`, run this:

```bash
git submodule update --init
```

2. Create a file named `.env` in the main project folder.

### 2. Diffbot Account

1. Create a free Diffbot account [here](https://www.diffbot.com/) with a work email address.
2. Copy the API Token.
3. Add the token as an environment variable in `.env`: `DIFFBOT_KEY=<YOUR_DIFFBOT_TOKEN>`

### 3. Neo4j AuraDB

1. Create a free Neo4j account [here](https://neo4j.com/product/auradb/).
2. Create a new AuraDB Free Instance.
3. When prompted, click "download". This downloads a `.txt` file with the credentials of your AuraDB instance.
4. Copy the variables in the `.txt` file into `.env`.

### 4. OpenAI

1. Create an OpenAI account [here](https://platform.openai.com).
2. Once logged in, click the settings icon on the top right.
3. In the left sidebar, click on **Billing** to access the billing settings.
4. Enter your payment information if you haven’t already set up a billing account. The minimum initial credit purchase is $5.
5. Create new secret key [here](https://platform.openai.com/account/api-keys).
6. A new API key will be generated for you. Copy this key.
7. Add the ley as an environment variable in `.env`: `OPENAI_KEY=<YOUR_OPENAI_KEY>`

### 5. Final `.env` file

Your `.env` file should have the following structure:

```bash
DIFFBOT_KEY=

NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
AURA_INSTANCEID=
AURA_INSTANCENAME=

OPENAI_KEY=
```

## Build and Run

### ER Extraction

```bash
docker compose -f compose.er-extraction.yaml up --build
```

This extracts entities and relationships using the Diffbot API and saves them into `output/nasdaq_kg_schema.json`. Since the Diffbot API has an API limit, we performed the extraction in batches. The complete output was saved in `output/merged_output.json`.

### KG Construction

```bash
docker compose -f compose.kg-construction.yaml up --build --abort-on-container-exit
```

This constructs the knowledge graph in the Neo4j Aura Instance from the nodes and relationships found in `output/merged_output.json`.

### KG Dashboard 
**Connect to Neo4j NeoDash**:
   1. Access NeoDash from a web browser at [https://neodash.graphapp.io](https://neodash.graphapp.io)
   2. Use the `neo4j+s` protocol for encrypted, secure connections.
   3. Provide the Aura database's URI, username, and password for authentication.
   4. On the left panel stated 'Dashboard', click create ("+") then "Import".
   5. Copy and paste the provided JSON (`dashboard.json`) into the "Import Dashboard" textbox. 
   6. Hit "Import" to load.
The imported dashboard will be automatically populated with predefined Cypher queries.You can now execute these queries to interact with your data and visualize your knowledge graph.

#### Additional Notes
- **Security**: Always ensure that your database credentials are kept secure and never share them publicly.
- **Customization**: You can modify the dashboard settings or Cypher queries as needed to fit your specific use case.
- **Troubleshooting**: If you encounter connection issues, verify your Neo4j Aura credentials and ensure your internet connection is stable.

### Chatbot

```bash
docker compose -f compose.chatbot.yaml up --build
```

This runs a streamlit app that is connected to the Neo4j Aura Instance.
Access the app at [http://localhost:8501/](http://localhost:8501/).

>>>>>>> Stashed changes
## Contributors

- Hui Qian
- Kaung Htet Wai Yan
- Tian Zhuoyu
- Chen Yuxi
- Tan Shu Hui (Amanda)
- Aidan Ong Zongren
- Xie Zebang
- Nguyen Xuan Nam
