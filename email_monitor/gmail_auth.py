"""
Gmail API OAuth 2.0 Authentication Module.
Enforces read-only least-privilege scope: https://www.googleapis.com/auth/gmail.readonly
"""

import os
from pathlib import Path
from typing import Optional

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
TOKEN_FILE = PROJECT_ROOT / "token.json"

def get_gmail_service():
    """
    Returns an authorized Gmail API service instance, or None if credentials are not configured.
    """
    if not CREDENTIALS_FILE.exists() and not TOKEN_FILE.exists():
        return None

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        if TOKEN_FILE.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            elif CREDENTIALS_FILE.exists():
                flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
                creds = flow.run_local_server(port=0)
            else:
                return None

            with open(TOKEN_FILE, "w", encoding="utf-8") as token:
                token.write(creds.to_json())

        service = build("gmail", "v1", credentials=creds)
        return service
    except Exception as e:
        print(f"[WARN] Failed to initialize Gmail service: {e}")
        return None
