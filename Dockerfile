FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements (if exists)
COPY requirements.txt .

# Install dependencies (Removed "|| true")
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (respecting .dockerignore)
COPY . .

# Expose FastAPI default port
EXPOSE 8002

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]