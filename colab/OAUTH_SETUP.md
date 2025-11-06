# Google OAuth Setup for Colab

## Why Use Your Own OAuth Credentials?

When using Colab's built-in `auth.authenticate_user()`, you get a "third party" warning because Colab acts as an intermediary. Using your own Google Cloud OAuth credentials eliminates this warning.

## Setup (One-Time)

### 1. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Project name: e.g., "NIH Chest X-Ray ML"

### 2. Enable Google Drive API

1. Go to **APIs & Services → Library**
2. Search for "Google Drive API"
3. Click **Enable**

### 3. Create OAuth Credentials

1. Go to **APIs & Services → Credentials**
2. Click **Create Credentials → OAuth client ID**
3. Application type: **Desktop app**
4. Name: e.g., "Colab Notebook"
5. Click **Create**

### 4. Download credentials.json

1. Click the download button (⬇️) next to your OAuth client
2. Rename to `client_secrets.json`
3. Save to `.colab/client_secrets.json` in your project

## File Format

Your `client_secrets.json` should look like this:

```json
{
  "installed": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uris": ["http://localhost"]
  }
}
```

## Using in Colab

The notebook will prompt you to upload `client_secrets.json` when you run Cell 2:

```python
# Cell 2 will show:
📁 Upload your client_secrets.json file
   (Get this from your local project: .colab/client_secrets.json)
```

1. Click the file upload button
2. Select your `client_secrets.json`
3. First run: OAuth consent screen will open
4. Click **Allow** to grant permissions
5. Credentials cached for future sessions

## Security Notes

✅ **Safe**:
- `client_secrets.json` contains OAuth credentials (not access tokens)
- Requires user consent before accessing any data
- Credentials are scoped to your Google Drive only

⚠️ **Keep Private**:
- Don't commit `client_secrets.json` to git (already in `.gitignore`)
- Don't share in public repositories
- Each developer should use their own credentials

## Troubleshooting

### "Access blocked: This app's request is invalid"

Your OAuth consent screen needs configuration:

1. Go to **APIs & Services → OAuth consent screen**
2. Select **External** (for personal use)
3. Fill in app name and your email
4. Add scope: `https://www.googleapis.com/auth/drive`
5. Save and continue
6. Try authenticating again

### "Invalid client_secrets.json"

Make sure:
- File is valid JSON
- Contains `"installed"` key (not `"web"`)
- Downloaded from correct OAuth client (Desktop app, not Web application)

### Credentials Not Persisting

PyDrive2 saves credentials to `credentials.json` in the current directory. This is ephemeral in Colab. To persist:

```python
# After first auth, upload credentials to Drive
upload_file_to_drive(drive, 'credentials.json', YOUR_FOLDER_ID)

# Next session, download before authenticating
# (Advanced - not implemented in default notebook)
```

## Alternative: Colab Built-in Auth

If you don't want to set up OAuth:

```python
# Replace Cell 2 with:
from google.colab import auth
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
```

⚠️ This will show "third party" warnings but works fine.

## References

- [Google Cloud OAuth 2.0](https://developers.google.com/identity/protocols/oauth2)
- [PyDrive2 Documentation](https://docs.iterative.ai/PyDrive2/)
- [Colab External Data Guide](https://colab.research.google.com/notebooks/io.ipynb)
