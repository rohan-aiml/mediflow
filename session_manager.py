"""
MediFlow AI — Session manager.
Pure asyncio — no threading. Designed for Sanic's single event loop.
"""

import asyncio
import json
import uuid
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator

from nova_sonic_client import BedrockStreamManager, SilentAudioStreamer
from clinical_extractor import ClinicalExtractor


@dataclass
class PatientInfo:
    name:     str = "Unknown"
    age:      str = ""
    gender:   str = ""
    mrn:      str = ""
    visit_id: str = field(default_factory=lambda: f"PT-{uuid.uuid4().hex[:8].upper()}")


@dataclass
class IntakeSession:
    session_id:   str
    patient:      PatientInfo
    created_at:   float = field(default_factory=time.time)
    is_active:    bool  = False
    is_complete:  bool  = False

    manager:         BedrockStreamManager | None = field(default=None, repr=False)
    silent_streamer: SilentAudioStreamer | None  = field(default=None, repr=False)
    extractor:       ClinicalExtractor           = field(default_factory=ClinicalExtractor, repr=False)

    # SSE queues — one per connected physician browser tab
    _sse_queues: list = field(default_factory=list, repr=False)

    # Text buffer for current AI turn
    _ai_buf:    str = field(default="", repr=False)


# ── Registry ───────────────────────────────────────────────────────────────────

_sessions: dict[str, IntakeSession] = {}


def create_session(patient: PatientInfo) -> IntakeSession:
    sid     = str(uuid.uuid4())
    session = IntakeSession(session_id=sid, patient=patient)
    _sessions[sid] = session
    return session


def get_session(sid: str) -> IntakeSession | None:
    return _sessions.get(sid)


def list_sessions() -> list[dict]:
    return [
        {
            "session_id": s.session_id,
            "patient":    s.patient.name,
            "mrn":        s.patient.mrn or s.patient.visit_id,
            "is_active":  s.is_active,
            "is_complete": s.is_complete,
            "severity":   s.extractor.note.severity.value,
            "created_at": s.created_at,
        }
        for s in _sessions.values()
    ]


# ── Lifecycle ──────────────────────────────────────────────────────────────────

async def start_session(session: IntakeSession, region: str = "us-east-1") -> None:
    """
    Initialise Bedrock stream, wire output subscriber, start silent heartbeat.
    Must be called from within Sanic's event loop (i.e. from a route handler).
    """
    session.manager = BedrockStreamManager(model_id="amazon.nova-2-sonic-v1:0", region=region)
    await session.manager.initialize_stream()

    # Open the first audio content block
    await session.manager.send_audio_content_start()

    # Start silent audio heartbeat (keeps connection alive before WS connects)
    session.silent_streamer = SilentAudioStreamer(session.manager)
    await session.silent_streamer.start()

    # Drain the output queue and dispatch events — background task
    asyncio.create_task(_drain_output(session))

    session.is_active = True
    await _broadcast(session, {"type": "session_started", "session_id": session.session_id})


async def stop_session(session: IntakeSession) -> None:
    session.is_active = False
    if session.silent_streamer:
        await session.silent_streamer.stop()
        session.silent_streamer = None
    if session.manager:
        await session.manager.close()
    session.is_complete = True
    await _broadcast(session, {"type": "session_complete", "soap": session.extractor.get_full_note()})


async def handoff_to_ws(session: IntakeSession) -> None:
    """
    Stop the silent streamer and open a fresh audio content block for real PCM.
    Must run on the same event loop as the session (Sanic's loop).
    """
    if session.silent_streamer:
        await session.silent_streamer.stop()
        session.silent_streamer = None
    # Rotate to a new audio_content_name — avoids double-contentStart rejection
    await session.manager.restart_audio_content()


# ── Audio / text input ─────────────────────────────────────────────────────────

def push_audio(session: IntakeSession, pcm: bytes) -> None:
    """
    Non-blocking. Called from Sanic's WS handler on the same event loop.
    put_nowait is safe here — no threading needed.
    """
    if session.manager and session.is_active:
        session.manager.add_audio_chunk(pcm)


async def send_patient_text(session: IntakeSession, text: str) -> None:
    if not session.manager or not session.is_active:
        return
    # Add to transcript, then ask Nova Lite to re-extract SOAP
    session.extractor.add_turn("PATIENT", text)
    soap = await session.extractor.extract_async()
    await _broadcast(session, {
        "type": "transcript", "role": "patient", "text": text,
        "soap": soap.to_dict(),
    })
    await session.manager.send_text(text)


# ── Output drain task ──────────────────────────────────────────────────────────

async def _drain_output(session: IntakeSession) -> None:
    """
    Background task: reads from manager.output_queue, processes events,
    and broadcasts to SSE queues. Runs until session ends.
    """
    manager = session.manager
    while session.is_active:
        try:
            data = await asyncio.wait_for(manager.output_queue.get(), timeout=1.0)
            await _handle_output(session, data)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[drain_output] error: {e}")


async def _handle_output(session: IntakeSession, event: dict) -> None:
    inner = event.get("event", {})

    if "textOutput" in inner:
        text = inner["textOutput"].get("content", "")
        if not text or '{ "interrupted" : true }' in text:
            return
        session._ai_buf += text
        await _broadcast(session, {"type": "ai_text_chunk", "text": text})

    elif "contentEnd" in inner:
        if session._ai_buf.strip():
            ai_text = session._ai_buf
            session._ai_buf = ""
            # Add AI turn, then re-extract full SOAP with Nova Lite
            session.extractor.add_turn("ASSISTANT", ai_text)
            soap = await session.extractor.extract_async()
            await _broadcast(session, {
                "type": "transcript",
                "role": "assistant",
                "text": ai_text,
                "soap": soap.to_dict(),
            })
            if soap.intake_complete:
                await stop_session(session)

    elif "audioOutput" in inner:
        # Audio bytes are already in manager.audio_output_queue
        # The WS handler drains that separately
        pass


# ── SSE ────────────────────────────────────────────────────────────────────────

async def _broadcast(session: IntakeSession, payload: dict) -> None:
    msg  = json.dumps(payload)
    dead = []
    for q in session._sse_queues:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        session._sse_queues.remove(q)


async def sse_generator(session: IntakeSession) -> AsyncGenerator[str, None]:
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    session._sse_queues.append(q)
    try:
        # Immediately push current state
        yield f"data: {json.dumps({'type':'snapshot','soap':session.extractor.note.to_dict(),'session_id':session.session_id})}\n\n"
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=15.0)
                yield f"data: {msg}\n\n"
                data = json.loads(msg)
                if data.get("type") in ("session_complete", "session_ended", "error"):
                    break
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    finally:
        if q in session._sse_queues:
            session._sse_queues.remove(q)
