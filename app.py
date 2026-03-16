"""
MediFlow AI — Sanic application.
Pure async — no Flask, no threading, no run_coroutine_threadsafe.
Sanic runs a single event loop; all session coroutines run on it natively.

Routes:
  POST /api/sessions              Create + start session
  GET  /api/sessions              List sessions
  GET  /api/sessions/<id>         Get session data
  DELETE /api/sessions/<id>       End session

  GET  /api/sessions/<id>/events  SSE → physician dashboard
  POST /api/sessions/<id>/text    Send patient text (testing)

  WS   /ws/<id>                   Full-duplex audio bridge

  GET  /                          Physician dashboard
  GET  /intake[/<id>]             Patient intake page
"""

import os
import asyncio
import json
import base64
from sanic import Sanic, Request, Websocket
from sanic.response import JSONResponse, HTTPResponse, text
from sanic.exceptions import SanicException
from dotenv import load_dotenv

load_dotenv()

from session_manager import (
    PatientInfo, create_session, get_session, list_sessions,
    start_session, stop_session, handoff_to_ws,
    push_audio, send_patient_text, sse_generator,
)

app = Sanic("mediflow")
app.config.WEBSOCKET_MAX_SIZE = 2 ** 20   # 1 MB frames
app.static("/static", "./static")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


# ── Helper ─────────────────────────────────────────────────────────────────────

def json_resp(data, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status=status)


# ── Session endpoints ──────────────────────────────────────────────────────────

@app.post("/api/sessions")
async def api_create_session(request: Request) -> JSONResponse:
    data    = request.json or {}
    patient = PatientInfo(
        name   = data.get("name", "Anonymous"),
        age    = data.get("age", ""),
        gender = data.get("gender", ""),
        mrn    = data.get("mrn", ""),
    )
    session = create_session(patient)
    # start_session runs on Sanic's event loop — no threading needed
    await start_session(session, region=AWS_REGION)
    return json_resp({"session_id": session.session_id, "visit_id": session.patient.visit_id}, 201)


@app.get("/api/sessions")
async def api_list_sessions(request: Request) -> JSONResponse:
    return json_resp(list_sessions())


@app.get("/api/sessions/<session_id>")
async def api_get_session(request: Request, session_id: str) -> JSONResponse:
    session = get_session(session_id)
    if not session:
        return json_resp({"error": "Not found"}, 404)
    return json_resp(session.extractor.get_full_note())


@app.delete("/api/sessions/<session_id>")
async def api_stop_session(request: Request, session_id: str) -> JSONResponse:
    session = get_session(session_id)
    if not session:
        return json_resp({"error": "Not found"}, 404)
    await stop_session(session)
    return json_resp({"status": "stopped"})


# ── SSE ────────────────────────────────────────────────────────────────────────

@app.get("/api/sessions/<session_id>/events")
async def api_sse_stream(request: Request, session_id: str):
    session = get_session(session_id)
    if not session:
        return json_resp({"error": "Not found"}, 404)

    response = await request.respond(
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    async for chunk in sse_generator(session):
        try:
            await response.write(chunk)
        except Exception:
            break
    return response


# ── Text input ─────────────────────────────────────────────────────────────────

@app.post("/api/sessions/<session_id>/text")
async def api_send_text(request: Request, session_id: str) -> JSONResponse:
    session = get_session(session_id)
    if not session:
        return json_resp({"error": "Not found"}, 404)
    data = request.json or {}
    text_input = (data.get("text") or "").strip()
    if not text_input:
        return json_resp({"error": "text required"}, 400)
    await send_patient_text(session, text_input)
    return json_resp({"status": "sent"})


# ── WebSocket audio bridge ─────────────────────────────────────────────────────

@app.websocket("/ws/<session_id>")
async def ws_audio(request: Request, ws: Websocket, session_id: str) -> None:
    """
    Full-duplex audio bridge — pure async, same event loop as everything else.

    Protocol:
      Client → Server: binary = raw Int16 PCM @ 16kHz
      Server → Client: binary = raw Int16 PCM @ 24kHz (AI audio)
      Server → Client: text   = JSON event (transcript, soap, alert)
    """
    session = get_session(session_id)
    if not session:
        await ws.send(json.dumps({"error": "Session not found"}))
        return

    manager = session.manager
    if not manager:
        await ws.send(json.dumps({"error": "Session not initialised"}))
        return

    # ── Hand off from silent streamer to real PCM ──────────────────────────────
    # This ends the old audio content block and opens a fresh one with a new UUID.
    # A second contentStart with the SAME name causes Bedrock to silently reject
    # all subsequent audio — the root cause of the previous "no response" bug.
    try:
        await handoff_to_ws(session)
    except Exception as e:
        await ws.send(json.dumps({"error": f"Handoff failed: {e}"}))
        return

    # Tell browser Bedrock is ready — it will start sending PCM after this
    await ws.send(json.dumps({"type": "ready"}))

    # ── Subscribe to SSE events — forward as text frames ──────────────────────
    sse_q: asyncio.Queue = asyncio.Queue(maxsize=128)
    session._sse_queues.append(sse_q)

    # ── Outbound audio pump — drains AI audio → binary WS frames ──────────────
    async def pump_audio():
        while session.is_active:
            try:
                chunk = await asyncio.wait_for(manager.audio_output_queue.get(), timeout=1.0)
                await ws.send(chunk)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

    # ── Outbound event pump — forwards JSON events as text WS frames ──────────
    async def pump_events():
        while session.is_active:
            try:
                msg = await asyncio.wait_for(sse_q.get(), timeout=1.0)
                await ws.send(msg)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

    audio_task  = asyncio.create_task(pump_audio())
    events_task = asyncio.create_task(pump_events())

    # ── Inbound — receive PCM from browser ────────────────────────────────────
    try:
        async for message in ws:
            if message is None:
                break
            if isinstance(message, bytes):
                # Raw Int16 PCM — push directly to audio queue (non-blocking)
                push_audio(session, message)
            elif isinstance(message, str):
                try:
                    cmd = json.loads(message)
                    if cmd.get("action") == "stop":
                        await stop_session(session)
                        break
                    elif cmd.get("action") == "text":
                        await send_patient_text(session, cmd.get("text", ""))
                except Exception:
                    pass
    finally:
        audio_task.cancel()
        events_task.cancel()
        await asyncio.gather(audio_task, events_task, return_exceptions=True)
        if sse_q in session._sse_queues:
            session._sse_queues.remove(sse_q)


# ── UI routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def physician_dashboard(request: Request) -> HTTPResponse:
    with open("templates/dashboard.html") as f:
        return HTTPResponse(f.read(), content_type="text/html")


@app.get("/intake/<session_id>")
async def patient_intake(request: Request, session_id: str = "") -> HTTPResponse:
    with open("templates/intake.html") as f:
        html = f.read().replace(
            '"{{ session_id | default(\'\') }}"',
            f'"{session_id}"'
        )
    return HTTPResponse(html, content_type="text/html")


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    return json_resp({"status": "ok", "service": "mediflow-ai"})


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5003))
    print("=" * 60)
    print("  MediFlow AI — Sanic Server")
    print("=" * 60)
    print(f"  Dashboard : http://localhost:{port}")
    print(f"  Intake    : http://localhost:{port}/intake")
    print("=" * 60)
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("\n  WARNING: AWS credentials not set\n")
    app.run(host="0.0.0.0", port=port, debug=False, single_process=True)
