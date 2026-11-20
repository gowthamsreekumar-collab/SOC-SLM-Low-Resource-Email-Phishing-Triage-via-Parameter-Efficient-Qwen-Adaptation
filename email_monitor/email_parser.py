"""
Email Preprocessing and Indicator Extraction Module.
Safely extracts headers, body, URLs, and attachment metadata from email messages
without executing attachments or clicking links.
"""

import email
import re
from email.header import decode_header
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

def decode_mime_header(header_value: Optional[str]) -> str:
    if not header_value:
        return ""
    decoded_parts = decode_header(header_value)
    result = []
    for part, enc in decoded_parts:
        if isinstance(part, bytes):
            result.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            result.append(str(part))
    return " ".join(result)

def extract_urls(text: str) -> List[str]:
    # Regex to extract HTTP/HTTPS URLs
    url_pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s()<>]+"
    raw_urls = re.findall(url_pattern, text)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in raw_urls:
        clean = u.rstrip(".,;!?'\")>]}")
        if clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped

def clean_html_body(html_content: str) -> str:
    # Strip script and style elements completely
    clean = re.sub(r"<script.*?>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<style.*?>.*?</style>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    # Convert <br> and <p> to newlines
    clean = re.sub(r"<br\s*/?>", "\n", clean, flags=re.IGNORECASE)
    clean = re.sub(r"</p>", "\n\n", clean, flags=re.IGNORECASE)
    # Strip remaining HTML tags
    clean = re.sub(r"<[^>]+>", " ", clean)
    # Normalize whitespaces
    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r"\n\s*\n", "\n\n", clean)
    return clean.strip()

def detect_sender_inconsistencies(sender_str: str) -> Dict[str, Any]:
    """
    Checks for display name spoofing (e.g. 'IT Support <hacker@external.com>').
    """
    match = re.search(r"^(.*?)\s*<([^>]+)>$", sender_str.strip())
    if not match:
        return {"display_name": sender_str, "email_address": sender_str, "spoof_suspect": False}

    display_name = match.group(1).strip().strip("\"'")
    email_address = match.group(2).strip()
    domain = email_address.split("@")[-1].lower() if "@" in email_address else ""

    # Known high-value impersonated brands
    target_brands = ["microsoft", "google", "apple", "paypal", "it support", "security", "admin", "hr", "payroll"]
    spoof_suspect = False
    reasons = []

    for b in target_brands:
        if b in display_name.lower() and b not in domain:
            spoof_suspect = True
            reasons.append(f"Display name references '{display_name}' but sender domain is '{domain}'")

    return {
        "display_name": display_name,
        "email_address": email_address,
        "domain": domain,
        "spoof_suspect": spoof_suspect,
        "reasons": reasons,
    }

def parse_raw_email(raw_content: str) -> Dict[str, Any]:
    msg = email.message_from_string(raw_content)

    subject = decode_mime_header(msg.get("Subject", "No Subject"))
    sender = decode_mime_header(msg.get("From", "Unknown"))
    date_str = decode_mime_header(msg.get("Date", ""))
    to_str = decode_mime_header(msg.get("To", ""))

    body_text = ""
    attachments_meta = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get("Content-Disposition", ""))

            if "attachment" in cdispo.lower():
                fname = decode_mime_header(part.get_filename() or "unnamed_attachment")
                payload = part.get_payload(decode=True) or b""
                attachments_meta.append({
                    "filename": fname,
                    "mime_type": ctype,
                    "size_bytes": len(payload),
                })
            elif ctype == "text/plain" and not body_text:
                payload = part.get_payload(decode=True)
                if payload:
                    body_text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            elif ctype == "text/html" and not body_text:
                payload = part.get_payload(decode=True)
                if payload:
                    html_str = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                    body_text = clean_html_body(html_str)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            raw_text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
            if msg.get_content_type() == "text/html":
                body_text = clean_html_body(raw_text)
            else:
                body_text = raw_text

    urls = extract_urls(body_text)
    sender_analysis = detect_sender_inconsistencies(sender)

    return {
        "subject": subject,
        "sender": sender,
        "date": date_str,
        "to": to_str,
        "body": body_text,
        "urls": urls,
        "attachments": attachments_meta,
        "sender_analysis": sender_analysis,
    }
