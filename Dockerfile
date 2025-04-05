# Use an official Python runtime as a parent image.
FROM python:3.8-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container.
COPY . /app

# Define the command to run the main script.
CMD ["python", "main.py"]
