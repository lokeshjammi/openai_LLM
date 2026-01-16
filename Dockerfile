# Use a lightweight Python image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your script into the container
COPY aws_bedrock_learning.py .

# Command to run the script
CMD ["python", "aws_bedrock_learning.py"]