# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY ./app ./app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable (optional, if you want to set a default for an API key, though .env or secrets are better for production)
# ENV ETHERSCAN_API_KEY YOUR_ETHERSCAN_API_KEY_HERE
# ENV OPENAI_API_KEY YOUR_OPENAI_API_KEY_HERE

# Run uvicorn server when the container launches
# The command needs to point to app.main:app if main.py is inside an 'app' subdirectory in the WORKDIR
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
