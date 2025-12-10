FROM python:3.10.13-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Railway provides PORT dynamically
ENV PORT=8501

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
