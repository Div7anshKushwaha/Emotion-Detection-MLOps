# Use the official lightweight Python 3.12 image as the base image
FROM python:3.12-slim

# Set the working directory inside the Docker container
WORKDIR /app

# Prevent Python from creating .pyc files
# Ensure Python output is displayed immediately in container logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy the Flask application's requirements file into the container
COPY flask_app/requirements.txt .

# Install all Python dependencies required by the Flask application
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application code into the container
COPY flask_app ./flask_app

# Copy the trained vectorizer and other model files into the container
COPY models ./models

# Download the NLTK resources required by the Flask application
RUN python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('stopwords', quiet=True)"

# Document that the Flask application listens on port 5000
EXPOSE 5000

# Command executed when the container starts
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "flask_app.app:app"]