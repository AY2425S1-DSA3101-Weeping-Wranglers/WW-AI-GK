# WW-AI-GK

This is the Weeping Wrangler's AI-Generated Knowledge Graph.

## Datasets

### Primary Datasets (Actual Data)

- **SEC 10-K Filings**: https://sec-api.io/resources/extract-textual-data-from-edgar-10-k-filings-using-python
  URLs of the top 40 NASDAQ companies were written to NASDAQ_10-K_urls.xlsx through referencing their CIK numbers, which were then used to extract 10-K filings using the SEC-API. The Form 1 and Form 7 sections were stored in a mySQL database and exported to sqlite3 file for persistent storage.
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

### Synthetic Data Generation

## Subtask 1: Entity and Relationship Extraction

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
