# MediFlow AI — Real-time Medical Intake Assistant

> Hackathon project built on Amazon Nova 2 Sonic bidirectional streaming.  
> Patients speak naturally; doctors receive a live SOAP note before they walk in.

---

## What it does

A patient arrives at a clinic. Instead of filling in a paper form or waiting for
a nurse, they speak to **MediFlow AI** — a voice agent powered by Amazon Nova 2
Sonic. The AI conducts a structured clinical intake interview (chief complaint →
HPI → PMH → medications → allergies → social history).

While the patient is speaking, the **physician dashboard** streams a live SOAP
note, computed risk scores (HEART score for chest pain), ICD-10 code suggestions,
and triage alerts. By the time the doctor walks in, they already know what they're
dealing with.

```
Patient speaks                     Doctor sees
──────────────                     ─────────────
"I have chest pain                 S: Chief complaint — chest tightness
 radiating to my jaw               O: PMH — HTN (self-reported)
 and I'm sweating"                 A: ACS pattern — HEART score 7/10
                                   P: ECG stat · Troponin · Aspirin 325mg
                  ◄── 300 ms ───►
```

---

## Architecture

```
Browser (patient)          Flask + flask-sock       Nova Sonic
────────────────           ─────────────────────    ───────────────────
AudioContext (16 kHz) ─── WS /ws/<sid> ─────────► BedrockStreamManager
                                                   (bidirectional stream)
                      ◄── PCM audio chunks ───────
                      ◄── JSON events ────────────

Browser (doctor)
────────────────
EventSource /api/sessions/<sid>/events
  ◄── SSE: transcript, soap_update, alert_flag
```

### Key files

| File | Purpose |
|---|---|
| `nova_sonic_client.py` | Bedrock bidirectional stream manager + silent audio streamer |
| `clinical_extractor.py` | Rule-based NLP → SOAP fields + HEART score + ICD-10 codes |
| `session_manager.py` | Session lifecycle, SSE broadcasting, audio routing |
| `app.py` | Flask REST API + WebSocket + SSE endpoints |
| `templates/intake.html` | Patient-facing voice UI |
| `templates/dashboard.html` | Physician real-time SOAP dashboard |

---

## Quick start

### 1. Clone and install

```bash
git clone <repo>
cd mediflow
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure AWS credentials

```bash
cp .env.example .env
# Edit .env — add AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
```

AWS permissions required:
- `bedrock:InvokeModelWithBidirectionalStream` on `amazon.nova-2-sonic-v1:0`

### 3. Run

```bash
python app.py
```

- Physician dashboard → http://localhost:5003  
- Patient intake      → http://localhost:5003/intake

### 4. Test without a microphone (text mode)

```bash
# Start a session
curl -X POST http://localhost:5003/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"name":"Test Patient","age":"38","gender":"M"}'

# Send text as the patient (use session_id from above)
curl -X POST http://localhost:5003/api/sessions/<session_id>/text \
  -H "Content-Type: application/json" \
  -d '{"text":"I have chest pain radiating to my jaw and left arm, started 4 hours ago, pain is 7 out of 10"}'

# Watch SSE events
curl -N http://localhost:5003/api/sessions/<session_id>/events
```

---

## API reference

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/sessions` | Create + start a session |
| `GET`  | `/api/sessions` | List all sessions |
| `GET`  | `/api/sessions/<id>` | Full session data + SOAP |
| `DELETE` | `/api/sessions/<id>` | End a session |

### Real-time

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/sessions/<id>/events` | SSE stream for physician dashboard |
| `WS`   | `/ws/<id>` | Full-duplex audio bridge |
| `POST` | `/api/sessions/<id>/text` | Send patient text (testing) |
| `POST` | `/api/sessions/<id>/audio` | Send PCM chunk (REST fallback) |
| `GET`  | `/api/sessions/<id>/audio` | Stream AI audio (chunked PCM) |

### SSE event types

```json
{"type": "snapshot",       "soap": {...}}
{"type": "transcript",     "role": "patient|assistant", "text": "...", "soap": {...}}
{"type": "ai_text_chunk",  "text": "..."}
{"type": "session_complete","soap": {"soap": {...}, "transcript": [...], "icd_codes": [...]}}
{"type": "error",          "message": "..."}
```

---
