FROM python:3.11-alpine
COPY requirements.txt .
COPY load_refdata.py .
COPY movie_poster.csv .
RUN pip install -r requirements.txt 
CMD python load_refdata.py
