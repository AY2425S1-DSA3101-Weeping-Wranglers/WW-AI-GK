FROM python:3.12.3

WORKDIR /app

ADD er_extraction/main_A.py .

RUN mkdir data
COPY data/ecmdatabase.db data/

COPY er_extraction/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "main_A.py"]