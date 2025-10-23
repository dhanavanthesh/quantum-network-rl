#!/bin/bash

echo "=============================================="
echo "Starting Quantum Network Simulator Backend"
echo "=============================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activating virtual environment..."
    if [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "ERROR: Virtual environment not found!"
        echo "Please run: python -m venv venv"
        exit 1
    fi
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing backend dependencies..."
    cd backend
    pip install -r requirements.txt
    cd ..
fi

echo ""
echo "Starting FastAPI server..."
echo "API will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the backend
cd backend
python main.py
