"""
Pydantic Schema and Output Validator for SOC Triage Assistant.
Ensures model responses strictly conform to the expected incident triage schema.
"""

import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

class AttackMappingItem(BaseModel):
    technique_id: str = Field(description="MITRE ATT&CK Technique ID, e.g. T1566.001")
    technique_name: str = Field(description="MITRE ATT&CK Technique Name, e.g. Spearphishing Attachment")

class IncidentTriageOutput(BaseModel):
    classification: str = Field(
        default="unknown",
        description="Threat classification: phishing | suspicious | benign | unknown"
    )
    severity: str = Field(
        default="low",
        description="Severity level: low | medium | high | critical | unknown"
    )
    summary: str = Field(
        default="",
        description="Concise summary of the email security assessment"
    )
    threat_score: int = Field(
        default=0,
        description="Calculated threat severity score from 0 (Benign) to 100 (Critical Malicious)"
    )
    confidence_score: float = Field(
        default=0.92,
        description="Model confidence level between 0.0 and 1.0"
    )
    threat_vectors: dict = Field(
        default_factory=lambda: {"sender_spoof": 10, "url_risk": 10, "urgency": 10, "evasion": 0},
        description="Sub-vector threat breakdowns"
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Specific indicators extracted from the email"
    )
    attack_mapping: List[AttackMappingItem] = Field(
        default_factory=list,
        description="Mapped MITRE ATT&CK techniques, if evidence justifies"
    )
    missing_evidence: List[str] = Field(
        default_factory=list,
        description="Information missing from the email needed for complete triage"
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended investigation and response steps for security analyst"
    )
    abstain: bool = Field(
        default=False,
        description="True if evidence is insufficient to make a confident assessment"
    )
    uncertainty_reason: str = Field(
        default="",
        description="Explanation when abstaining or when confidence is low"
    )
    analyst_review_required: bool = Field(
        default=True,
        description="Always true: system acts as analyst support, requiring human review"
    )


def extract_and_validate_json(raw_text: str) -> tuple[IncidentTriageOutput, bool, str]:
    """
    Extracts JSON from model output, validates with Pydantic, and returns
    (validated_object, is_valid, raw_or_error_string).
    """
    cleaned = raw_text.strip()

    # 1. Direct parse
    parsed_dict = None
    try:
        parsed_dict = json.loads(cleaned)
    except Exception:
        pass

    # 2. Markdown fenced code blocks
    if parsed_dict is None:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            try:
                parsed_dict = json.loads(match.group(1))
            except Exception:
                pass

    # 3. First '{' to last '}'
    if parsed_dict is None:
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                parsed_dict = json.loads(match.group(1))
            except Exception:
                pass

    if parsed_dict is None:
        # Fallback heuristic
        fallback = IncidentTriageOutput(
            classification="unknown",
            severity="medium",
            summary="Failed to parse structured JSON from model response.",
            evidence=[],
            attack_mapping=[],
            missing_evidence=["Valid structured model response"],
            recommended_actions=["Manual analyst inspection required."],
            abstain=True,
            uncertainty_reason="Model generated malformed non-JSON output.",
            analyst_review_required=True,
        )
        return fallback, False, "JSON extraction failed"

    # Normalize classifications
    valid_classes = {"phishing", "suspicious", "benign", "unknown"}
    if "classification" in parsed_dict and parsed_dict["classification"].lower() in valid_classes:
        parsed_dict["classification"] = parsed_dict["classification"].lower()
    else:
        parsed_dict["classification"] = "unknown"

    valid_sevs = {"low", "medium", "high", "critical", "unknown"}
    if "severity" in parsed_dict and parsed_dict["severity"].lower() in valid_sevs:
        parsed_dict["severity"] = parsed_dict["severity"].lower()
    else:
        parsed_dict["severity"] = "low"

    # Normalize attack mapping if strings were returned instead of dicts
    if "attack_mapping" in parsed_dict and isinstance(parsed_dict["attack_mapping"], list):
        normalized_mappings = []
        for item in parsed_dict["attack_mapping"]:
            if isinstance(item, dict) and "technique_id" in item:
                normalized_mappings.append(item)
            elif isinstance(item, str):
                normalized_mappings.append({"technique_id": item, "technique_name": item})
        parsed_dict["attack_mapping"] = normalized_mappings

    try:
        validated = IncidentTriageOutput.model_validate(parsed_dict)
        return validated, True, ""
    except ValidationError as e:
        # Graceful correction
        fallback = IncidentTriageOutput(
            classification=str(parsed_dict.get("classification", "unknown")),
            severity=str(parsed_dict.get("severity", "medium")),
            summary=str(parsed_dict.get("summary", "Validation error occurred.")),
            evidence=list(parsed_dict.get("evidence", [])) if isinstance(parsed_dict.get("evidence"), list) else [],
            recommended_actions=list(parsed_dict.get("recommended_actions", [])) if isinstance(parsed_dict.get("recommended_actions"), list) else ["Manual analyst verification"],
            abstain=bool(parsed_dict.get("abstain", True)),
            uncertainty_reason=f"Pydantic validation warning: {e}",
            analyst_review_required=True,
        )
        return fallback, False, str(e)
