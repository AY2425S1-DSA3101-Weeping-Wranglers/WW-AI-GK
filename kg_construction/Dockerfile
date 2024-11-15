FROM python:3.11.6-slim

WORKDIR /app

# Install system dependencies required for pyreadr
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    r-base \
    && rm -rf /var/lib/apt/lists/*

COPY kg_construction/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir kg_construction
COPY kg_construction/main_B_12.py kg_construction/
COPY kg_construction/fact_checking.py kg_construction/
COPY kg_construction/graph_utils.py kg_construction/

RUN mkdir data
COPY data/UNSD_m49.csv data/
COPY data/country_aliases.csv data/
COPY data/corp_tax_rate.xlsx data/

WORKDIR /app/kg_construction

CMD ["python", "-u", "main_B_12.py"]