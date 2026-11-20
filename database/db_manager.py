"""
SQLite Database Manager for SOC Triage Assistant.
Stores incident logs, extracted metadata, triage assessments, and analyst reviews.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("database/soc_triage.db")

def init_db(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email_id TEXT UNIQUE,
        sender TEXT,
        subject TEXT,
        date TEXT,
        body TEXT,
        urls TEXT,
        model_variant TEXT,
        classification TEXT,
        severity TEXT,
        summary TEXT,
        evidence TEXT,
        attack_mapping TEXT,
        missing_evidence TEXT,
        recommended_actions TEXT,
        abstain INTEGER,
        uncertainty_reason TEXT,
        analyst_review_required INTEGER DEFAULT 1,
        analyst_feedback TEXT DEFAULT 'pending',
        analyst_notes TEXT DEFAULT '',
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_incident(incident_data: Dict[str, Any], db_path: Path = DB_PATH) -> int:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    email_id = incident_data.get("email_id", f"sim_{int(datetime.utcnow().timestamp())}")
    sender = incident_data.get("sender", "Unknown")
    subject = incident_data.get("subject", "No Subject")
    date_str = incident_data.get("date", datetime.utcnow().isoformat())
    body = incident_data.get("body", "")
    urls = json.dumps(incident_data.get("urls", []))
    model_variant = incident_data.get("model_variant", "LoRA")

    assessment = incident_data.get("assessment", {})
    classification = assessment.get("classification", "unknown")
    severity = assessment.get("severity", "low")
    summary = assessment.get("summary", "")
    evidence = json.dumps(assessment.get("evidence", []))
    attack_mapping = json.dumps(assessment.get("attack_mapping", []))
    missing_evidence = json.dumps(assessment.get("missing_evidence", []))
    recommended_actions = json.dumps(assessment.get("recommended_actions", []))
    abstain = 1 if assessment.get("abstain", False) else 0
    uncertainty_reason = assessment.get("uncertainty_reason", "")
    created_at = datetime.utcnow().isoformat()

    cursor.execute("""
    INSERT OR REPLACE INTO incidents (
        email_id, sender, subject, date, body, urls, model_variant,
        classification, severity, summary, evidence, attack_mapping,
        missing_evidence, recommended_actions, abstain, uncertainty_reason,
        created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        email_id, sender, subject, date_str, body, urls, model_variant,
        classification, severity, summary, evidence, attack_mapping,
        missing_evidence, recommended_actions, abstain, uncertainty_reason,
        created_at
    ))

    incident_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return incident_id

def get_all_incidents(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM incidents ORDER BY id DESC")
    rows = cursor.fetchall()

    incidents = []
    for r in rows:
        item = dict(r)
        item["urls"] = json.loads(item["urls"]) if item["urls"] else []
        item["evidence"] = json.loads(item["evidence"]) if item["evidence"] else []
        item["attack_mapping"] = json.loads(item["attack_mapping"]) if item["attack_mapping"] else []
        item["missing_evidence"] = json.loads(item["missing_evidence"]) if item["missing_evidence"] else []
        item["recommended_actions"] = json.loads(item["recommended_actions"]) if item["recommended_actions"] else []
        item["abstain"] = bool(item["abstain"])
        incidents.append(item)

    conn.close()
    return incidents

def update_analyst_feedback(incident_id: int, feedback: str, notes: str = "", db_path: Path = DB_PATH):
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE incidents
    SET analyst_feedback = ?, analyst_notes = ?
    WHERE id = ?
    """, (feedback, notes, incident_id))

    conn.commit()
    conn.close()
