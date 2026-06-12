#!/bin/bash

# SmartFarm AI Environment Setup Script

echo "=== SmartFarm AI: Environment Setup ==="

# Check for python3
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 could not be found. Please install Python 3."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment (venv)..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies from smart-farm/requirements.txt..."
pip install -r smart-farm/requirements.txt

echo "=== Setup Complete ==="
echo "To start the server, run: source venv/bin/activate && cd smart-farm && python3 main.py"
echo "To run the similarity check, run: source venv/bin/activate && cd smart-farm && python3 match.py"
