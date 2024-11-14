# WW-AI-GK

This is the Weeping Wranglers' AI-Generated Knowledge Graph for Enterprise Content Management (ECM) Systems.

The NASDAQ exchange, which tracks over 3,000 stocks with a focus on tech and high-growth sectors, presents a complex challenge for investors performing due diligence. The current process of scrutinizing financial statements, news articles, and public data sources demands significant time and resources from investors and venture capital firms. Our knowledge graph solution addresses this challenge by integrating these diverse data sources into a unified, queryable format that enables rapid analysis and decision-making.

Through generating the knowledge graph, we hope to support investors by providing a comprehensive view of a company's ecosystem, partners, and competitors, which can serve as a holistic basis to assess a company’s fundamental value, sphere of influence, and growth potential.

## Datasets

- **SEC 10-K Filings**: https://sec-api.io/resources/extract-textual-data-from-edgar-10-k-filings-using-python
  URLs of the top 40 NASDAQ companies were written to NASDAQ_10-K_urls.xlsx through referencing their CIK numbers, which were then used to extract 10-K filings using the SEC-API. The Form 1 and Form 7 sections were stored in a mySQL database and exported to sqlite3 file for persistent storage.
  **Wikipedia Articles** Using the yahooquery library, the official websites of companies are retrieved and used to find their related Wikipedia pages, extracting relevant information using Beautiful Soup's HTML parser.
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

Entity resolution can be performed during this stage via **the ReFiNED Transformer**, which links entity mentions in documents to their corresponding Wikidata entries from a database of over 30 million entities, or during knowledge graph construction through by using vector indexes to disambiguate entities whose vectors have high pairwise similarity scores. 

To scale up our ER Extraction Pipeline, we can build a distributed processing architecture that begins with **data ingestion into distributed storage systems** through the Hadoop Distributed File System or Amazon S3. These systems provide redundancy and fault tolerance for our large-scale unstructured data. We process this data using the **Spark DataFrame API**, which distributes aggregation tasks across multiple worker nodes for parallel processing. 
We **batch our API requests** to minimize network overhead and maintain a **caching system** for frequently accessed nodes and queries. Our caching system can maintain frequently accessed data in memory using a Least Recently Used policy, which significantly reduces database load and improves response times.
Third, we can achieve effective resource management through **integrating our Spark API with YARN**, which dynamically allocates resources based on current workload demands. Data can also be **partitioned**, segmenting information by company, region, and industry to ensure balanced workload distribution. 


## Subtask 2: Knowledge Graph Construction and Verification

## Contributors

- Hui Qian
- Kaung Htet Wai Yan
- Tian Zhuoyu
- Chen Yuxi
- Tan Shu Hui (Amanda)
- Aidan Ong Zongren
- Xie Zebang
- Nguyen Xuan Nam
