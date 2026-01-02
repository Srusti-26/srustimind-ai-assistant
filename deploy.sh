#!/bin/bash
# Production deployment script

echo "Starting deployment..."

# Install dependencies
pip install -r requirements.txt

# Set production environment
export FLASK_ENV=production
export FLASK_DEBUG=False

# Start with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app