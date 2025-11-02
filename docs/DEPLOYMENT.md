# Deployment Guide - NIH Chest X-Ray Disease Detection

This guide covers deploying the Streamlit dashboard to Streamlit Cloud.

## Prerequisites

- GitHub repository with your project
- Streamlit Cloud account (https://streamlit.io/cloud)
- Python 3.11 (configured via `.python-version`)

## Streamlit Cloud Deployment

### Initial Setup (First Time Only)

1. **Sign in to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app" button
   - Select your GitHub repository: `manwithacat/CapStone`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: Choose your custom subdomain (e.g., `nihxrays.streamlit.app`)

3. **Advanced Settings** (Optional)
   - Python version: `3.11` (auto-detected from `.python-version`)
   - Click "Deploy"

### Automatic Deployment on Push

Once your app is connected to Streamlit Cloud:

✅ **Automatic deployment is already configured!**

Every time you push to the `main` branch on GitHub:
1. Streamlit Cloud detects the push
2. Rebuilds the app with updated code
3. Redeploys automatically (usually takes 2-5 minutes)

**To trigger a deployment:**
```bash
git add .
git commit -m "your message"
git push origin main
```

**To manually restart without code changes:**
- Go to Streamlit Cloud dashboard
- Click "⋮" menu on your app
- Select "Reboot app"

### Configuration Files

- **`.streamlit/config.toml`**: Server and theme configuration (committed to git)
- **`.streamlit/secrets.toml`**: Sensitive credentials (NOT in git, configure in Streamlit Cloud)
- **`.python-version`**: Python version (3.11)
- **`requirements.txt`**: Production dependencies

### Managing Secrets (If Needed)

Currently, this app doesn't require secrets. If you need to add secrets in the future:

1. Go to your app on Streamlit Cloud
2. Click "⋮" menu → "Settings"
3. Go to "Secrets" section
4. Add secrets in TOML format:
   ```toml
   [kaggle]
   username = "your_username"
   key = "your_api_key"
   ```
5. Click "Save"

Access secrets in code:
```python
import streamlit as st
kaggle_user = st.secrets["kaggle"]["username"]
```

### Monitoring Deployment

**Check deployment status:**
1. Go to https://share.streamlit.io/
2. Find your app in the dashboard
3. View logs by clicking on the app

**Common deployment issues:**
- **Module not found**: Check `requirements.txt` includes all dependencies
- **Python version mismatch**: Verify `.python-version` is `3.11`
- **Memory errors**: Large datasets may need optimization or caching

### App URL

Your deployed app will be available at:
```
https://nihxrays.streamlit.app
```
(or your chosen subdomain)

### Updating the App

**For code changes:**
```bash
# Make your changes
git add .
git commit -m "feat: add new feature"
git push origin main
# App redeploys automatically in 2-5 minutes
```

**For dependency changes:**
```bash
# Update requirements.txt
git add requirements.txt
git commit -m "build: update dependencies"
git push origin main
# Full rebuild triggered (may take 5-10 minutes)
```

**For configuration changes:**
```bash
# Update .streamlit/config.toml
git add .streamlit/config.toml
git commit -m "config: update theme"
git push origin main
# App restarts with new config
```

### Performance Optimization

**Caching data:**
```python
@st.cache_data
def load_data():
    # Load expensive data once
    return pd.read_csv("data.csv")
```

**Caching resources:**
```python
@st.cache_resource
def load_model():
    # Load ML model once
    return tf.keras.models.load_model("model.h5")
```

### Troubleshooting

**App won't start:**
1. Check GitHub repository is public or Streamlit Cloud has access
2. Verify `app.py` exists in root directory
3. Check requirements.txt for syntax errors
4. Review logs in Streamlit Cloud dashboard

**Deployment slow:**
- Large dependencies (TensorFlow) take time to install
- First deployment takes longest (5-10 minutes)
- Subsequent deployments are faster (2-5 minutes)

**Memory issues:**
- Streamlit Cloud free tier has 1GB RAM limit
- Optimize data loading with caching
- Consider loading data on-demand rather than at startup

### Local Testing Before Deployment

Always test locally first:
```bash
# Activate virtual environment
source .venv/bin/activate

# Run app locally
streamlit run app.py

# Test in browser at http://localhost:8501
```

### Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Deploy your app](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [App management](https://docs.streamlit.io/streamlit-community-cloud/manage-your-app)
- [Troubleshooting](https://docs.streamlit.io/streamlit-community-cloud/troubleshooting)

## Alternative Deployment Options

### Heroku (Not Recommended for This Project)

This project was previously configured for Heroku but those files have been removed. Streamlit Cloud is the recommended platform as it's:
- Free for public repositories
- Optimized for Streamlit apps
- Automatic deployment on git push
- Better resource allocation for data science apps

### Docker (Advanced)

For custom deployment environments:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

## Security Considerations

- ✅ Secrets excluded from git (`.streamlit/secrets.toml` in `.gitignore`)
- ✅ Large data files excluded (via `.gitignore`)
- ✅ API credentials managed through Streamlit Cloud secrets
- ⚠️ App is publicly accessible - don't include real patient data
- ⚠️ Research and educational use only (not clinical deployment)

## Support

For deployment issues:
- Check Streamlit Community Forum: https://discuss.streamlit.io/
- Review GitHub Issues for this project
- Contact Code Institute support for assessment-related questions
