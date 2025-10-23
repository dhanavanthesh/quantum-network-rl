# Complete Setup Commands - Step by Step

## ✅ Fixed Commands (Use These!)

### Method 1: Easy Start (Recommended)

```bash
# Navigate to project directory
cd C:\Users\Dhana\Desktop\quantum\q

# Activate virtual environment
source venv/Scripts/activate  # Git Bash
# OR
venv\Scripts\activate  # CMD

# Start backend server (from project root, NOT backend folder!)
python start_server.py
```

**That's it!** The server will start on http://localhost:8000

---

### Method 2: Using Startup Scripts

#### For Git Bash (Windows):
```bash
cd C:\Users\Dhana\Desktop\quantum\q
chmod +x start_backend.sh
./start_backend.sh
```

#### For Windows CMD/PowerShell:
```cmd
cd C:\Users\Dhana\Desktop\quantum\q
start_backend.bat
```

---

### Method 3: Manual Commands (If you prefer)

```bash
# Navigate to project ROOT (not backend folder!)
cd C:\Users\Dhana\Desktop\quantum\q

# Activate venv
source venv/Scripts/activate

# Start server from project root
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🚀 Complete First-Time Setup

### Step 1: Create Virtual Environment

```bash
cd C:\Users\Dhana\Desktop\quantum\q
python -m venv venv
```

### Step 2: Activate Virtual Environment

**Git Bash:**
```bash
source venv/Scripts/activate
```

**CMD:**
```cmd
venv\Scripts\activate
```

**PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```

You should see `(venv)` in your prompt:
```
(venv) Dhana@DKING MINGW64 ~/Desktop/quantum/q
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
cd backend
pip install -r requirements.txt
cd ..
```

### Step 4: Start Backend Server

**Option A - Easy launcher:**
```bash
python start_server.py
```

**Option B - Direct uvicorn:**
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 5: Verify Server is Running

Open another terminal and test:
```bash
curl http://localhost:8000/health
```

Or open in browser: http://localhost:8000/docs

---

## 🎨 Frontend Setup (Separate Terminal)

**Open a NEW terminal** (keep backend running in the first one)

```bash
# Navigate to frontend folder
cd C:\Users\Dhana\Desktop\quantum\q\frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

Frontend will be at: http://localhost:5173

---

## 🧪 Testing Without Web Interface

If you just want to run simulations without the web UI:

```bash
cd C:\Users\Dhana\Desktop\quantum\q
source venv/Scripts/activate

# Quick test
python run_simulation.py --mode quick --n-nodes 50 --n-episodes 100

# With custom settings
python run_simulation.py \
  --mode quick \
  --n-nodes 100 \
  --topology grid \
  --n-temporal 32 \
  --n-spectral 16 \
  --n-episodes 200 \
  --output-dir results
```

---

## 🐛 Troubleshooting

### Error: "attempted relative import with no known parent package"

**Solution:** Don't run `python main.py` from the backend folder. Instead:

```bash
# Go to project ROOT
cd C:\Users\Dhana\Desktop\quantum\q

# Run from root
python start_server.py
```

### Error: "No module named 'backend'"

**Solution:** Make sure you're in the project root directory:

```bash
pwd  # Should show: /c/Users/Dhana/Desktop/quantum/q
ls   # Should show: backend, frontend, venv, etc.

# If you're in the wrong directory:
cd C:\Users\Dhana\Desktop\quantum\q
```

### Error: "venv not activated"

**Solution:**
```bash
# Check if activated (should see (venv) in prompt)
source venv/Scripts/activate  # Git Bash
# OR
venv\Scripts\activate  # CMD
```

### Error: "ModuleNotFoundError: No module named 'fastapi'"

**Solution:** Install dependencies:
```bash
source venv/Scripts/activate
pip install -r backend/requirements.txt
```

### Error: Port 8000 already in use

**Solution:** Use different port:
```bash
python -m uvicorn backend.main:app --reload --port 8001
```

---

## 📋 Quick Reference

### Start Backend (Choose ONE method)

```bash
# Method 1: Python launcher (easiest)
python start_server.py

# Method 2: Bash script
./start_backend.sh

# Method 3: Batch script (Windows)
start_backend.bat

# Method 4: Direct uvicorn
python -m uvicorn backend.main:app --reload
```

### Start Frontend

```bash
cd frontend
npm run dev
```

### Run Simulation (No Web UI)

```bash
python run_simulation.py --mode quick --n-nodes 50 --n-episodes 150
```

### Run Examples

```bash
python example_usage.py
```

---

## ✅ Verification Checklist

After setup, verify everything works:

```bash
# 1. Check Python version
python --version  # Should be 3.9+

# 2. Check venv is activated
which python  # Should point to venv

# 3. Check backend dependencies
python -c "import fastapi, torch, networkx; print('✓ OK')"

# 4. Start backend
python start_server.py

# 5. In another terminal, test API
curl http://localhost:8000/health

# 6. Open in browser
# Visit: http://localhost:8000/docs
```

---

## 🎯 Two-Terminal Workflow

### Terminal 1 - Backend
```bash
cd C:\Users\Dhana\Desktop\quantum\q
source venv/Scripts/activate
python start_server.py
```

### Terminal 2 - Frontend
```bash
cd C:\Users\Dhana\Desktop\quantum\q\frontend
npm run dev
```

**Access:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🚀 You're Ready!

Now you can:
1. ✅ Run simulations via command line
2. ✅ Use the web interface
3. ✅ Access the REST API
4. ✅ Run example scripts

Happy simulating! 🎉
