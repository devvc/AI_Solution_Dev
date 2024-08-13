# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the preprocessing script
COPY data_preprocessing.py .

# Command to run the preprocessing script
CMD ["python", "data_preprocessing.py"]
