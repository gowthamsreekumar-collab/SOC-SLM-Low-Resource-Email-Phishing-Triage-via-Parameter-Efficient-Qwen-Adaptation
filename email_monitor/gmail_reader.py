"""
Gmail Reader Module.
Polls for incoming emails using Gmail API, extracts raw MIME messages,
and feeds them to the email parser.
"""

import base64
from typing import Any, Dict, List, Optional
from email_monitor.email_parser import parse_raw_email
from email_monitor.gmail_auth import get_gmail_service

def fetch_recent_emails(max_results: int = 10, label_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    service = get_gmail_service()
    if not service:
        return []

    if label_ids is None:
        label_ids = ["INBOX"]

    try:
        results = service.users().messages().list(
            userId="me",
            labelIds=label_ids,
            maxResults=max_results
        ).execute()

        messages = results.get("messages", [])
        parsed_emails = []

        for msg_meta in messages:
            msg_id = msg_meta["id"]
            raw_msg = service.users().messages().get(
                userId="me",
                id=msg_id,
                format="raw"
            ).execute()

            raw_bytes = base64.urlsafe_b64decode(raw_msg["raw"].encode("ASCII"))
            raw_str = raw_bytes.decode("utf-8", errors="replace")

            parsed = parse_raw_email(raw_str)
            parsed["email_id"] = msg_id
            parsed_emails.append(parsed)

        return parsed_emails
    except Exception as e:
        print(f"[ERROR] Failed to fetch emails from Gmail: {e}")
        return []
