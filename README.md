# WW-AI-GK

This is the Weeping Wranglers' AI-Generated Knowledge Graph for Enterprise Content Management (ECM) Systems.

The NASDAQ exchange, which tracks over 3,000 stocks with a focus on tech and high-growth sectors, presents a complex challenge for investors performing due diligence. The current process of scrutinizing financial statements, news articles, and public data sources demands significant time and resources from investors and venture capital firms. Our knowledge graph solution addresses this challenge by integrating these diverse data sources into a unified, queryable format that enables rapid analysis and decision-making.

Through generating the knowledge graph, we hope to support investors by providing a comprehensive view of a company's ecosystem, partners, and competitors, which can serve as a holistic basis to assess a company’s fundamental value, sphere of influence, and growth potential.

<p align="center">
   <a href="https://www.youtube.com/watch?v=ZJv7Ch9aOY0">
      <img src="http://img.youtube.com/vi/ZJv7Ch9aOY0/0.jpg" alt="Video Presentation">
   </a>
</p>

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
Note: You may skip this step if you do not intend to use the Chatbot feature.

1. Create an OpenAI account [here](https://platform.openai.com).
2. Once logged in, click the settings icon on the top right.
3. In the left sidebar, click on Billing to access the billing settings.
4. Enter your payment information if you haven’t already set up a billing account. The minimum initial credit purchase is $5.
5. Create new secret key [here](https://platform.openai.com/account/api-keys).
6. A new API key will be generated for you. Copy this key.
7. Add the key as an environment variable in `.env`: `OPENAI_KEY=<YOUR_OPENAI_KEY>`

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

Check that your AuraDB instance has been initialised and is running from [here](console.neo4j.io), before running this command:

```bash
docker compose -f compose.kg-construction.yaml up --build --abort-on-container-exit
```

This constructs the knowledge graph in the Neo4j Aura Instance from the nodes and relationships found in `output/merged_output.json`.

### KG Dashboard 
**Connect to Neo4j NeoDash**:
   1. Access NeoDash from a web browser at [https://neodash.graphapp.io](https://neodash.graphapp.io).
   2. Click "New Dashboard"
   3. Use the `neo4j+s` protocol.
   4. For Hostname, enter the part after "neo4j+s://" of your NEO4J_URI (saved in your `.env` file). For example if your NEO4J_URI is "neo4j+s://abcde123.databases.neo4j.io", your Hostname will be "abcde123.databases.neo4j.io".
   5. For Password, enter your NEO4J_PASSWORD (saved in your `.env` file).
   6. Click Connect.
   7. On the left panel, click create ("+") then "Import".
   8. Import the provided JSON (Found in `chatbot/neodash_dashboard.json`) into the "Import Dashboard" textbox. 
   9. Hit "Import" to load.
The imported dashboard will be automatically populated with predefined Cypher queries. You can now execute these queries to interact with your data and visualize your knowledge graph. Feel free to create your own dashboard, tailored to your business needs.

### Streamlit App

```bash
docker compose -f compose.chatbot.yaml up --build
```

This runs a streamlit app that is connected to the Neo4j Aura Instance. Requires an OpenAI API key if you intend to use the chatbot feature.
Access the app at [http://localhost:8501/](http://localhost:8501/).


## Contributors

- Hui Qian
- Kaung Htet Wai Yan
- Tian Zhuoyu
- Chen Yuxi
- Tan Shu Hui (Amanda)
- Aidan Ong Zongren
- Xie Zebang
- Nguyen Xuan Nam
