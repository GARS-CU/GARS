# Start with a base image that includes Python and pip
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install OS dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install Python dependencies
RUN pip install --no-cache-dir numpy tensorflow keras scikit-learn matplotlib opencv-python-headless flask Pillow

# # Expose port 5000 for the Flask app
# EXPOSE 5000

# Your application's entry point
CMD ["python", "engagement_preprocess.py"]