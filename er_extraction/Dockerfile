FROM python:3.12.3

WORKDIR /app

ADD main_A.py .

RUN mkdir data
COPY data/ecmdatabase.db data/

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main_A.py"]