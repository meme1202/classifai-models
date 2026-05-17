FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application including models and frontend files
COPY . .

# Expose port
EXPOSE 7860

# Run with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
