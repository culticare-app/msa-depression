# Create a new build stage from a base image
FROM python:3.12-slim

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Change working directory
WORKDIR /app

# Copy necessary files and directories, including the model file
COPY requirements.txt .
COPY app.py .
COPY .env .
COPY bert_emotion_model.pth .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Describe which ports your application is listening on
EXPOSE 8000

# Start the application
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
