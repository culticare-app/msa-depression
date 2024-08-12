# Create a new build stage from a base image
FROM python:3.12-slim

# Change working directory
WORKDIR /app

# Copy necessary files and directories (excluding the model file)
COPY requirements.txt .
COPY app.py .
COPY .env .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Describe which ports your application is listening on
EXPOSE 8000

# Download the model file and start the application
ENTRYPOINT ["/bin/sh", "-c", "curl -L -o /app/bert_emotion_model.pth 'https://drive.google.com/uc?id=1--02mYsEKB_3PV5LN7t5rFcKIunNA0Mq' && uvicorn app:app --host 0.0.0.0 --port 8000"]
