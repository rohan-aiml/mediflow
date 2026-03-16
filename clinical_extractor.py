"""
MediFlow AI — AI-powered clinical extractor.

Instead of brittle regex, we send the full conversation transcript to
Amazon Nova Lite (fast, cheap, 128k context) via the Bedrock converse API.

Nova Lite returns a structured JSON SOAP note.  We call it:
  - After every completed assistant turn (incremental update)
  - On demand via get_full_note()

This runs as a standard asyncio coroutine on Sanic's event loop.
"""

import asyncio
import json
import os
import boto3
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


class TriageSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MODERATE = "MODERATE"
    LOW      = "LOW"
    UNKNOWN  = "UNKNOWN"


@dataclass
class SOAPNote:
    # S — Subjective
    chief_complaint:  str  = ""
    onset:            str  = ""
    duration:         str  = ""
    pain_scale:       str  = ""
    radiation:        str  = ""
    aggravating:      str  = ""
    relieving:        str  = ""
    associated_sx:    list = field(default_factory=list)
    # O — Objective (voiced)
    pmh:              list = field(default_factory=list)
    medications:      list = field(default_factory=list)
    allergies:        str  = "Not yet reported"
    social_hx:        str  = ""
    family_hx:        str  = ""
    # A — Assessment
    primary_ddx:      str  = ""
    secondary_ddx:    list = field(default_factory=list)
    risk_flags:       list = field(default_factory=list)
    heart_score:      Optional[int] = None
    # P — Plan
    immediate_actions: list = field(default_factory=list)
    suggested_meds:    list = field(default_factory=list)
    consults:          list = field(default_factory=list)
    disposition:       str  = ""
    # Meta
    severity:         TriageSeverity = TriageSeverity.UNKNOWN
    confidence:       int  = 0
    intake_complete:  bool = False
    alert_flags:      list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SOAPNote":
        d = dict(d)
        sev = d.pop("severity", "UNKNOWN")
        note = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        try:
            note.severity = TriageSeverity(sev)
        except ValueError:
            note.severity = TriageSeverity.UNKNOWN
        return note


# ── Prompt sent to Nova Lite ───────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a clinical NLP engine. You receive a medical intake conversation
and return ONLY a valid JSON object — no markdown, no explanation, no preamble.

Extract every clinically relevant detail the patient mentioned and fill the JSON schema below.
For fields not mentioned, use empty string "" or empty array [].

TRIAGE RULES (fill alert_flags and severity accordingly):
- If chest pain + radiation to arm/jaw/neck + diaphoresis/nausea → severity=CRITICAL, primary_ddx="Acute Coronary Syndrome (r/o STEMI)", alert_flags=["ACS triad — escalate immediately","Door-to-balloon target <90 min"], heart_score = numeric 1-10 estimate
- If facial droop / arm weakness / slurred speech → severity=CRITICAL, primary_ddx="Acute ischaemic stroke (r/o haemorrhagic)", alert_flags=["Activate stroke protocol"]
- If throat swelling / hives / hypotension → severity=CRITICAL, primary_ddx="Anaphylaxis"
- If chest pain only (no triad) → severity=HIGH
- If general complaint, no red flags → severity=LOW or MODERATE

JSON SCHEMA (return exactly this structure):
{
  "chief_complaint": "",
  "onset": "",
  "duration": "",
  "pain_scale": "",
  "radiation": "",
  "aggravating": "",
  "relieving": "",
  "associated_sx": [],
  "pmh": [],
  "medications": [],
  "allergies": "",
  "social_hx": "",
  "family_hx": "",
  "primary_ddx": "",
  "secondary_ddx": [],
  "risk_flags": [],
  "heart_score": null,
  "immediate_actions": [],
  "suggested_meds": [],
  "consults": [],
  "disposition": "",
  "severity": "UNKNOWN",
  "confidence": 0,
  "intake_complete": false,
  "alert_flags": []
}

confidence: integer 0-100 reflecting how much clinical data has been gathered.
intake_complete: true ONLY if the AI said "doctor will be with you shortly" or similar."""


def _build_transcript_text(transcript: list[dict]) -> str:
    lines = []
    for t in transcript:
        role  = t.get("role", "").upper()
        label = "Patient" if role in ("PATIENT", "USER") else "MediFlow AI"
        lines.append(f"{label}: {t.get('text','')}")
    return "\n".join(lines) if lines else "(no conversation yet)"


# ── Extractor ──────────────────────────────────────────────────────────────────

class ClinicalExtractor:
    """
    Uses Amazon Nova Lite (bedrock converse API) to extract structured SOAP
    from the conversation transcript after each completed AI turn.
    Falls back to the last known note if the API call fails.
    """

    MODEL_ID = os.getenv("EXTRACTION_MODEL", "amazon.nova-lite-v1:0")

    def __init__(self):
        self.note        = SOAPNote()
        self._transcript: list[dict] = []
        self._client     = None   # boto3 bedrock-runtime, lazily created
        self._lock       = asyncio.Lock()   # one extraction at a time

    def _get_client(self):
        if self._client is None:
            region = os.getenv("AWS_REGION", "us-east-1")
            self._client = boto3.client("bedrock-runtime", region_name=region)
        return self._client

    # ── Called from session_manager after each completed turn ─────────────────

    def add_turn(self, role: str, text: str) -> None:
        """Append a turn to the transcript (synchronous — just appends)."""
        self._transcript.append({"role": role, "text": text})
        # Mark intake_complete immediately if AI said the magic phrase
        if role.upper() in ("ASSISTANT", "AI"):
            if "doctor will be with you" in text.lower() or "everything i need" in text.lower():
                self.note.intake_complete = True

    async def extract_async(self) -> SOAPNote:
        """
        Call Nova Lite to re-extract the full SOAP note from current transcript.
        Non-blocking — runs on Sanic's event loop via run_in_executor.
        Returns the updated SOAPNote.
        """
        if not self._transcript:
            return self.note

        async with self._lock:
            transcript_text = _build_transcript_text(self._transcript)
            try:
                note = await asyncio.get_event_loop().run_in_executor(
                    None, self._call_nova, transcript_text
                )
                # Preserve intake_complete if we already set it
                if self.note.intake_complete:
                    note.intake_complete = True
                self.note = note
            except Exception as e:
                print(f"[ClinicalExtractor] Nova Lite error: {e}")
                # Keep last known note — don't wipe it on transient failure
            return self.note

    def _call_nova(self, transcript_text: str) -> SOAPNote:
        """
        Synchronous Bedrock converse call — runs in executor thread.
        Uses boto3 (not the async Bedrock SDK) for simplicity.
        """
        client = self._get_client()
        response = client.converse(
            modelId=self.MODEL_ID,
            system=[{"text": EXTRACTION_SYSTEM}],
            messages=[{
                "role": "user",
                "content": [{
                    "text": (
                        "Extract a SOAP note from this medical intake conversation.\n\n"
                        f"{transcript_text}\n\n"
                        "Return ONLY the JSON object."
                    )
                }]
            }],
            inferenceConfig={
                "maxTokens": 1024,
                "temperature": 0.1,   # low temp for structured extraction
                "topP": 0.9,
            },
        )

        raw_text = response["output"]["message"]["content"][0]["text"].strip()

        # Strip markdown fences if model wraps in ```json ... ```
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        data = json.loads(raw_text)
        return SOAPNote.from_dict(data)

    # ── Public helpers ─────────────────────────────────────────────────────────

    def get_full_note(self) -> dict:
        return {
            "soap":       self.note.to_dict(),
            "transcript": self._transcript,
        }
