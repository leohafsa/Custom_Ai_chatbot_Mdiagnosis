# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app.py requirements.txt /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# Expose the port your FastAPI application will run on
EXPOSE 3012

CMD [ "python3", "-m" , "flask", "run","--host=0.0.0.0", "--port=3012"]
