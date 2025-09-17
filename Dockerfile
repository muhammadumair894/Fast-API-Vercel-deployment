# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy requirements into the container first for caching layer benefits
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI code into the container
COPY main.py .

# Expose port 8000 for the API
EXPOSE 8000

# Run the FastAPI app with uvicorn when the container launches
CMD [ "uvicorn", "rag_fastapi_endpoints:app", "--host", "0.0.0.0", "--port", "8000" ]
