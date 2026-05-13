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
```

## Connecting Frontend to Backend

The HTML automatically detects the backend URL:
- **Local:** `http://localhost:5000`
- **Production:** Railway deployment URL (set in `getBackendUrl()` function)

To set backend URL dynamically:
```javascript
localStorage.setItem('backendUrl', 'https://your-railway-url.railway.app');
```

## CORS Configuration

Both frontend and backend support CORS:
- Backend (`edge/roadai.py`): Allows all origins with `@app.after_request` decorator
- Frontend: Uses dynamic backend URL to avoid cross-origin issues

## Troubleshooting

### Video stream not loading
- Check if Railway backend is running: `https://<app>.railway.app/health`
- Verify backend URL in browser console: `console.log(BACKEND_URL)`
- Ensure CORS headers are being sent

### Railway deployment fails
- Check buildpack configuration in `railway.toml`
- Verify `requirements.txt` has all dependencies
- Check Railway logs: Dashboard → Logs

### Environment variables not working
- Ensure variables are set in Railway dashboard
- Restart deployment after changing variables
- Check `.env` file exists locally for development

## File Structure
```
road-ai/
├── edge/
│   └── roadai.py          # Flask backend (deployed to Railway)
├── index.html             # Frontend (deployed to GitHub Pages)
├── requirements.txt       # Python dependencies
├── railway.toml          # Railway deployment config
├── .env.example          # Environment variables template
└── README_DEPLOYMENT.md  # This file
```
