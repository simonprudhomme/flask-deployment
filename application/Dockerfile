# Base image
FROM python:3.6-slim-buster

# Copy contents
COPY . /app

# Change work directory
WORKDIR /app

# Install the requirements
RUN apt-get update && apt-get install build-essential -y
RUN pip install -r requirements.txt

# Start the application
CMD ["python", "app.py"]