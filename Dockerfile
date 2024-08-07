FROM python:3.9-slim  # Change to Python 3.9

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Command to run the app
CMD ["uvicorn", "apps.src.app:app", "--host", "0.0.0.0", "--port", "8000"]
