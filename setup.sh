#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements/dev.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "PYTHONPATH=${PYTHONPATH}:${PWD}" > .env
fi

# Add .env loading to activate script
if ! grep -q "source .env" venv/bin/activate; then
    echo -e "\n# Load .env file\nsource .env" >> venv/bin/activate
fi

echo "Setup completed! To activate the environment, run: source venv/bin/activate" 