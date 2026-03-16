"""
MediFlow AI — Nova Sonic client.
Modeled directly on the reference asyncio implementation (no RxPY).
Key design decisions from reference:
  - asyncio.Queue for audio I/O (not RxPY Subject)
  - _process_audio_input runs as its own asyncio Task
  - add_audio_chunk uses put_nowait (non-blocking, callable from any context)
  - All coroutines run on ONE event loop (Sanic's loop) — no threading
"""

import asyncio
import base64
import json
import uuid
import warnings
import datetime
import inspect
from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
)
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

warnings.filterwarnings("ignore")

INPUT_SAMPLE_RATE  = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS           = 1
CHUNK_SIZE         = 1024

DEBUG = False


def debug_print(msg: str) -> None:
    if DEBUG:
        caller = inspect.stack()[1].function
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] [{caller}] {msg}")


# ── Medical intake system prompt ───────────────────────────────────────────────

MEDICAL_SYSTEM_PROMPT = (
    "You are MediFlow, an AI medical intake specialist. "
    "Conduct a structured pre-triage clinical interview gathering: "
    "chief complaint, HPI (onset, severity 1-10, radiation, aggravating/relieving), "
    "associated symptoms, PMH, medications, allergies, social history. "
    "Ask ONE focused question at a time. Be warm and calm. Never diagnose aloud. "
    "If patient describes chest pain + radiation + diaphoresis, say: "
    "'I'm flagging this for the doctor immediately.' "
    "Keep responses under 2 sentences. "
    "After gathering sufficient data say exactly: "
    "'Thank you. I have everything I need — the doctor will be with you shortly.'"
)


# ── BedrockStreamManager ───────────────────────────────────────────────────────

class BedrockStreamManager:
    """
    Pure-asyncio Nova Sonic stream manager.
    All methods are coroutines that run on Sanic's event loop.
    No threads, no RxPY, no run_coroutine_threadsafe.
    """

    # ── Event templates ────────────────────────────────────────────────────────

    START_SESSION_EVENT = json.dumps({
        "event": {
            "sessionStart": {
                "inferenceConfiguration": {
                    "maxTokens": 512,
                    "topP": 0.85,
                    "temperature": 0.5,
                }
            }
        }
    })

    @property
    def PROMPT_START_EVENT(self) -> str:
        return json.dumps({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": OUTPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "tiffany",
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                    "toolConfiguration": {"tools": []},
                }
            }
        })

    def _content_start_audio(self) -> str:
        return json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": INPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    },
                }
            }
        })

    def _text_content_start(self, content_name: str, role: str, interactive: bool = False) -> str:
        return json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "role": role,
                    "type": "TEXT",
                    "interactive": interactive,
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        })

    def _text_input(self, content_name: str, text: str) -> str:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        return json.dumps({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": escaped,
                }
            }
        })

    def _audio_input(self, b64: str) -> str:
        return json.dumps({
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": b64,
                }
            }
        })

    def _content_end(self, content_name: str) -> str:
        return json.dumps({
            "event": {"contentEnd": {"promptName": self.prompt_name, "contentName": content_name}}
        })

    def _prompt_end(self) -> str:
        return json.dumps({
            "event": {"promptEnd": {"promptName": self.prompt_name}}
        })

    SESSION_END_EVENT = json.dumps({"event": {"sessionEnd": {}}})

    # ── Init ───────────────────────────────────────────────────────────────────

    def __init__(self, model_id: str = "amazon.nova-2-sonic-v1:0", region: str = "us-east-1"):
        self.model_id  = model_id
        self.region    = region
        self.is_active = False
        self.barge_in  = False

        # Queues — asyncio-native, no threads needed
        self.audio_input_queue:  asyncio.Queue = asyncio.Queue()
        self.audio_output_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue:       asyncio.Queue = asyncio.Queue()

        self.bedrock_client  = None
        self.stream_response = None

        # Tasks
        self._response_task:    asyncio.Task | None = None
        self._audio_input_task: asyncio.Task | None = None

        # Session IDs
        self.prompt_name        = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self._system_content_name = str(uuid.uuid4())

        # State
        self.role = None
        self.display_assistant_text = False

    def _init_client(self) -> None:
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)

    async def initialize_stream(self) -> "BedrockStreamManager":
        if not self.bedrock_client:
            self._init_client()

        self.stream_response = await self.bedrock_client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True

        # Send init sequence exactly as reference does: small delay between events
        sys_cname = self._system_content_name
        init_events = [
            self.START_SESSION_EVENT,
            self.PROMPT_START_EVENT,
            self._text_content_start(sys_cname, "SYSTEM", interactive=False),
            self._text_input(sys_cname, MEDICAL_SYSTEM_PROMPT),
            self._content_end(sys_cname),
        ]
        for ev in init_events:
            await self._send_raw(ev)
            await asyncio.sleep(0.05)   # small gap between init events (from reference)

        # Start background tasks (same pattern as reference)
        self._response_task    = asyncio.create_task(self._process_responses())
        self._audio_input_task = asyncio.create_task(self._process_audio_input())

        await asyncio.sleep(0.1)   # let tasks start
        debug_print("Stream initialised")
        return self

    # ── Send ───────────────────────────────────────────────────────────────────

    async def _send_raw(self, event_json: str) -> None:
        if not self.stream_response or not self.is_active:
            return
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        try:
            await self.stream_response.input_stream.send(chunk)
            debug_print(f"sent {list(json.loads(event_json).get('event',{}).keys())}")
        except Exception as e:
            debug_print(f"send error: {e}")

    async def send_audio_content_start(self) -> None:
        await self._send_raw(self._content_start_audio())

    async def send_audio_content_end(self) -> None:
        await self._send_raw(self._content_end(self.audio_content_name))

    async def restart_audio_content(self) -> None:
        """
        Rotate to a fresh audio content block.
        Called when WS client takes over from the silent heartbeat.
        Sending a second contentStart with the same name → Bedrock rejects silently.
        """
        await self.send_audio_content_end()
        self.audio_content_name = str(uuid.uuid4())
        await asyncio.sleep(0.05)
        await self.send_audio_content_start()
        debug_print(f"audio content restarted → {self.audio_content_name}")

    async def send_text(self, text: str) -> None:
        """Send patient text input (text-mode / testing)."""
        cname = str(uuid.uuid4())
        await self._send_raw(self._text_content_start(cname, "USER", interactive=True))
        await self._send_raw(self._text_input(cname, text))
        await self._send_raw(self._content_end(cname))

    # ── Audio input — asyncio.Queue pattern from reference ────────────────────

    def add_audio_chunk(self, pcm_bytes: bytes) -> None:
        """
        Non-blocking. Called from Sanic's WS handler (same event loop).
        Uses put_nowait exactly as reference does — no threading needed.
        """
        self.audio_input_queue.put_nowait({"audio_bytes": pcm_bytes})

    async def _process_audio_input(self) -> None:
        """
        Dedicated task: drains audio_input_queue → base64-encodes → sends to Bedrock.
        Identical structure to reference _process_audio_input.
        """
        while self.is_active:
            try:
                data = await self.audio_input_queue.get()
                audio_bytes = data.get("audio_bytes")
                if not audio_bytes:
                    continue
                b64 = base64.b64encode(audio_bytes).decode("utf-8")
                await self._send_raw(self._audio_input(b64))
            except asyncio.CancelledError:
                break
            except Exception as e:
                debug_print(f"audio input error: {e}")

    # ── Response processing ────────────────────────────────────────────────────

    async def _process_responses(self) -> None:
        """
        Dedicated task: reads from Bedrock stream, puts events on output_queue
        and audio_output_queue. Identical structure to reference.
        """
        try:
            while self.is_active:
                try:
                    output = await self.stream_response.await_output()
                    result = await output[1].receive()
                    if result.value and result.value.bytes_:
                        raw  = result.value.bytes_.decode("utf-8")
                        data = json.loads(raw)
                        await self._dispatch(data)
                        await self.output_queue.put(data)
                except StopAsyncIteration:
                    break
                except Exception as e:
                    if "ValidationException" in str(e):
                        print(f"Bedrock validation error: {e}")
                    else:
                        debug_print(f"response error: {e}")
                    break
        except Exception as e:
            debug_print(f"response task error: {e}")
        finally:
            self.is_active = False

    async def _dispatch(self, data: dict) -> None:
        event = data.get("event", {})
        if "contentStart" in event:
            self.role = event["contentStart"].get("role")
            fields = event["contentStart"].get("additionalModelFields", "{}")
            try:
                parsed = json.loads(fields)
                self.display_assistant_text = parsed.get("generationStage") == "SPECULATIVE"
            except Exception:
                pass
        elif "textOutput" in event:
            text = event["textOutput"].get("content", "")
            if '{ "interrupted" : true }' in text:
                self.barge_in = True
        elif "audioOutput" in event:
            audio_bytes = base64.b64decode(event["audioOutput"]["content"])
            await self.audio_output_queue.put(audio_bytes)

    # ── Teardown ───────────────────────────────────────────────────────────────

    async def close(self) -> None:
        if not self.is_active:
            return
        if self._audio_input_task and not self._audio_input_task.done():
            self._audio_input_task.cancel()
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
        await self.send_audio_content_end()
        await self._send_raw(self._prompt_end())
        await self._send_raw(self.SESSION_END_EVENT)
        self.is_active = False
        try:
            await self.stream_response.input_stream.close()
        except Exception:
            pass
        debug_print("stream closed")


# ── Silent audio heartbeat (keeps Bedrock alive between patient turns) ─────────

class SilentAudioStreamer:
    """
    Sends silent PCM chunks to keep the Nova Sonic session alive.
    Uses put_nowait (same as reference) — no blocking.
    """
    SILENT_CHUNK = b"\x00" * (CHUNK_SIZE * 2)

    def __init__(self, manager: BedrockStreamManager):
        self.manager    = manager
        self.is_running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self.is_running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self.is_running:
            self.manager.add_audio_chunk(self.SILENT_CHUNK)
            await asyncio.sleep(0.02)
