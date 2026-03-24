FROM python:3.11-slim
 
WORKDIR /app
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY src/ ./src/
COPY tenders_index.json ./tenders_index.json
COPY .env.example .env.example
 
ENV PYTHONUNBUFFERED=1
 
CMD ["python", "src/mcp_server.py"]
 
