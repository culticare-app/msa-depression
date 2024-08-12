# Create a new build stage from a base image
FROM python:3.12-slim

# Change working directory
WORKDIR /app

# Copy necessary files and directories
COPY requirements.txt .
COPY app.py .
COPY .env .
COPY bert_emotion_model.pth .

# Execute build commands
RUN pip install --no-cache-dir -r requirements.txt

# Describe which ports your application is listening on
EXPOSE 8000

# Specify default executable
ENTRYPOINT [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000" ]
