# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Need to be accessed externally
EXPOSE 5000

# Copy the inference script & html files
COPY model_inference.py .
COPY templates/ templates/

# Command to run the preprocessing script
CMD ["python", "model_inference.py"]
