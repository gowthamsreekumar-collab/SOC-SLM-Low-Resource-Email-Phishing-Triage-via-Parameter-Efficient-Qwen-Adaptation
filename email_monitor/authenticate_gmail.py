"""
One-Time Gmail Authentication Setup Script.
Prompts OAuth consent in your web browser and stores token.json locally.
"""

import sys
from pathlib import Path

# Add project root to sys.path so imports work regardless of how it's executed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from email_monitor.gmail_auth import CREDENTIALS_FILE, TOKEN_FILE, get_gmail_service


def main():
    print("=" * 60)
    print("  GMAIL API AUTHENTICATION SETUP")
    print("=" * 60)

    if not CREDENTIALS_FILE.exists():
        print(f"\n[ERROR] '{CREDENTIALS_FILE}' not found in the project root.")
        print("Please ensure credentials.json is placed in:")
        print(f"  {PROJECT_ROOT / 'credentials.json'}")
        return

    print("\n[INFO] Found 'credentials.json'. Launching browser consent flow...")
    print("A browser window will open. Please approve the read-only Gmail access.\n")

    service = get_gmail_service()
    if service and TOKEN_FILE.exists():
        print(f"\n[SUCCESS] Authentication successful!")
        print(f"Saved persistent credentials token to: {TOKEN_FILE.resolve()}")
        print("\nThe Streamlit dashboard is now authorized to pull real inbox emails!")
    else:
        print("\n[FAILED] Authentication failed. Please verify credentials.json configuration.")


if __name__ == "__main__":
    main()
