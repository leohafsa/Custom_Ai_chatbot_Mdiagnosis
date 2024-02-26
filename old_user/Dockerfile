# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /ap
COPY app.py util.py requirements.txt ./app/
COPY .  /app/  


# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# Expose the port your FastAPI application will run on
EXPOSE 3012

CMD [ "python", "app.py" ]