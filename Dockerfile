# Create a new build stage from a base image
FROM python:3.12-slim

# Change working directory
WORKDIR /app

# Copy necessary files and directories (except the model file)
COPY requirements.txt .
COPY app.py .
COPY .env .

# Execute build commands
RUN pip install --no-cache-dir -r requirements.txt

# Describe which ports your application is listening on
EXPOSE 8000

# Download model file and start the application
ENTRYPOINT ["sh", "-c", "if [ ! -f /app/bert_emotion_model.pth ]; then curl -L -o /app/bert_emotion_model.pth https://your-storage-service.com/path/to/bert_emotion_model.pth; fi && uvicorn app:app --host 0.0.0.0 --port 8000"]
