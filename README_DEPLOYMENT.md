# Deployment Guide: GitHub Pages + Railway

## Overview
This project uses:
- **GitHub Pages** for the static HTML frontend (webapp)
- **Railway** for the Python Flask backend (edge inference)

## Frontend Deployment (GitHub Pages)

1. **Enable GitHub Pages:**
   - Go to repo Settings → Pages
   - Select "Deploy from a branch"
   - Choose `main` branch and `/root` directory
   - Your site will be available at: `https://1nkst.github.io/road-ai/`

2. **Update Backend URL in HTML:**
   - Open `index.html`
   - Find the `getBackendUrl()` function
   - Replace `'https://your-railway-app.railway.app'` with your actual Railway URL
   - Or use `localStorage.setItem('backendUrl', 'YOUR_RAILWAY_URL')` in browser console

## Backend Deployment (Railway)

1. **Connect Repository to Railway:**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select `1nkst/road-ai` repository
   - Railway will auto-detect the Python app

2. **Set Environment Variables in Railway:**
   - API_URL: Your Roboflow API endpoint
   - API_KEY: Your Roboflow API key
   - WORKSPACE: Your Roboflow workspace name
   - WORKFLOW_ID: Your Roboflow workflow ID
   - PORT: 5000 (automatically set)

3. **Deploy:**
   - Railway will automatically deploy on every push to `main`
   - Access your API at: `https://<your-railway-app>.railway.app`

## Testing Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set up .env file (copy from .env.example)
cp .env.example .env

# Run the Flask backend
python edge/roadai.py

# Open index.html in browser
# Backend URL will default to http://localhost:5000