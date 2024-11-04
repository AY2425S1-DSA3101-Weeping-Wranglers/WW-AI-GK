FROM python:3.12.3

ADD main_A.py .
ADD ecmdatabase.db .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./main_A.py"]