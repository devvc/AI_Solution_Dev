# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model training script
COPY model_training.py .

# Command to run the model training script
CMD ["python", "model_training.py"]
