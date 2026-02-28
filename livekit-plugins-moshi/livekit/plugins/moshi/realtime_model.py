from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterable, Awaitable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents.llm.chat_context import ChatContext, FunctionCall
from livekit.agents.llm.realtime import (
    EventTypes,
    GenerationCreatedEvent,
    InputSpeechStartedEvent,
    InputSpeechStoppedEvent,
    MessageGeneration,
    RealtimeCapabilities,
    RealtimeModel,
    RealtimeModelError,
    RealtimeSession,
)
from livekit.agents.llm.tool_context import Tool, ToolChoice, ToolContext
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .log import logger

try:
    import sphn
except ImportError:
    raise ImportError(
        "sphn is required for the Moshi plugin. Install with: pip install 'sphn>=0.1.4'"
    )

MOSHI_SAMPLE_RATE = 24000


def _shortuuid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


@dataclass
class MoshiOptions:
    server_url: str
    voice: str
    text_prompt: str
    text_temperature: float
    text_topk: int
    audio_temperature: float
    audio_topk: int
    seed: int


class MoshiRealtimeModel(RealtimeModel):
    """LiveKit RealtimeModel implementation for Moshi/PersonaPlex.

    Connects to a running Moshi WebSocket server and bridges audio
    between LiveKit's PCM AudioFrames and Moshi's Opus-encoded protocol.

    Args:
        server_url: WebSocket URL of the Moshi server (e.g. "ws://localhost:8998")
        voice: Voice prompt filename (e.g. "NATM1", "NATF2", "VARM0")
        text_prompt: System prompt for the model's persona/role
        text_temperature: Sampling temperature for text tokens
        text_topk: Top-k sampling for text tokens
        audio_temperature: Sampling temperature for audio tokens
        audio_topk: Top-k sampling for audio tokens
        seed: Random seed (-1 to disable)
    """

    def __init__(
        self,
        *,
        server_url: str = "ws://localhost:8998",
        voice: str = "NATM1",
        text_prompt: str = "",
        text_temperature: float = 0.7,
        text_topk: int = 25,
        audio_temperature: float = 0.8,
        audio_topk: int = 250,
        seed: int = -1,
    ):
        super().__init__(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,  # Moshi handles turn-taking internally
                user_transcription=True,  # Moshi emits text tokens
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
            )
        )
        self._opts = MoshiOptions(
            server_url=server_url,
            voice=voice,
            text_prompt=text_prompt,
            text_temperature=text_temperature,
            text_topk=text_topk,
            audio_temperature=audio_temperature,
            audio_topk=audio_topk,
            seed=seed,
        )

    @property
    def model(self) -> str:
        return "nvidia/personaplex-7b-v1"

    @property
    def provider(self) -> str:
        return "moshi"

    def session(self) -> MoshiRealtimeSession:
        return MoshiRealtimeSession(self, self._opts)

    async def aclose(self) -> None:
        pass


class MoshiRealtimeSession(RealtimeSession):
    """Manages a single bidirectional audio session with a Moshi server.

    Handles:
    - WebSocket connection and Moshi handshake protocol
    - PCM <-> Opus codec bridging via sphn
    - Sample rate conversion (48kHz from LiveKit -> 24kHz for Moshi)
    - Streaming audio/text output as LiveKit GenerationCreatedEvent
    """

    def __init__(self, model: MoshiRealtimeModel, opts: MoshiOptions):
        super().__init__(model)
        self._opts = opts
        self._chat_ctx = ChatContext()
        self._tools = ToolContext([])

        # Audio input queue: PCM AudioFrames from LiveKit -> Moshi
        self._audio_input_queue: asyncio.Queue[rtc.AudioFrame] = asyncio.Queue(
            maxsize=200
        )

        # Audio/text output channels: Moshi -> LiveKit
        # These are replaced each time a new generation starts
        self._audio_output_queue: asyncio.Queue[rtc.AudioFrame | None] = (
            asyncio.Queue()
        )
        self._text_output_queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Opus codec instances for bridging PCM <-> Opus
        self._opus_writer: sphn.OpusStreamWriter | None = None
        self._opus_reader: sphn.OpusStreamReader | None = None

        # Connection state
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._connected = asyncio.Event()
        self._closed = False
        self._current_generation_id: str | None = None
        self._pending_generation_fut: asyncio.Future[GenerationCreatedEvent] | None = (
            None
        )

        # Start connection in background
        self._main_task = asyncio.ensure_future(self._run())

    # -- Required properties --

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> ToolContext:
        return self._tools

    # -- Context management (limited support for Moshi) --

    async def update_instructions(self, instructions: str) -> None:
        logger.debug(
            "update_instructions called but Moshi does not support "
            "mid-session instruction updates"
        )

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        self._chat_ctx = chat_ctx

    async def update_tools(self, tools: list[Tool]) -> None:
        logger.debug("update_tools called but Moshi does not support tools")

    def update_options(
        self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN
    ) -> None:
        pass

    # -- Audio I/O --

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Accept a PCM AudioFrame from LiveKit and queue it for Moshi."""
        if self._closed:
            return
        try:
            self._audio_input_queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # Drop frame if backed up

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass  # Moshi is audio-only

    # -- Generation control --

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        """Request the model to generate a reply.

        For Moshi, this creates a new generation event. Since Moshi is
        full-duplex and always generating, this effectively starts a new
        output stream segment.
        """
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[GenerationCreatedEvent] = loop.create_future()

        if self._connected.is_set():
            self._start_new_generation(fut, user_initiated=True)
        else:
            # Store and resolve after connection
            self._pending_generation_fut = fut

        return fut

    def commit_audio(self) -> None:
        pass  # Moshi processes audio continuously

    def clear_audio(self) -> None:
        # Drain pending input frames
        while not self._audio_input_queue.empty():
            try:
                self._audio_input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def interrupt(self) -> None:
        # Moshi is full-duplex and handles interruptions natively.
        # When the user speaks, the model naturally adjusts.
        # We signal the current generation's streams to end so LiveKit
        # can start consuming a fresh generation.
        if self._current_generation_id is not None:
            self._end_current_generation()

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass  # Moshi doesn't support truncation

    # -- Lifecycle --

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True

        # End current generation streams
        self._end_current_generation()

        if self._ws and not self._ws.closed:
            await self._ws.close()

        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass

    # -- Internal: generation management --

    def _start_new_generation(
        self,
        fut: asyncio.Future[GenerationCreatedEvent] | None = None,
        user_initiated: bool = False,
    ) -> None:
        """Create a new generation with fresh audio/text output streams."""
        gen_id = _shortuuid("gen_")
        msg_id = _shortuuid("msg_")
        self._current_generation_id = gen_id

        # Create fresh output queues
        audio_queue: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
        text_queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Signal end of previous streams
        old_audio = self._audio_output_queue
        old_text = self._text_output_queue
        old_audio.put_nowait(None)
        old_text.put_nowait(None)

        # Install new queues
        self._audio_output_queue = audio_queue
        self._text_output_queue = text_queue

        # Build async iterables
        async def audio_stream() -> AsyncIterable[rtc.AudioFrame]:
            while True:
                frame = await audio_queue.get()
                if frame is None:
                    break
                yield frame

        async def text_stream() -> AsyncIterable[str]:
            while True:
                text = await text_queue.get()
                if text is None:
                    break
                yield text

        # Modalities future (immediately resolved)
        modalities_fut: asyncio.Future[list[Literal["text", "audio"]]] = (
            asyncio.get_event_loop().create_future()
        )
        modalities_fut.set_result(["text", "audio"])

        msg_gen = MessageGeneration(
            message_id=msg_id,
            text_stream=text_stream(),
            audio_stream=audio_stream(),
            modalities=modalities_fut,
        )

        async def message_stream() -> AsyncIterable[MessageGeneration]:
            yield msg_gen

        async def function_stream() -> AsyncIterable[FunctionCall]:
            return
            yield  # type: ignore  # make this an async generator

        event = GenerationCreatedEvent(
            message_stream=message_stream(),
            function_stream=function_stream(),
            user_initiated=user_initiated,
            response_id=gen_id,
        )

        if fut is not None and not fut.done():
            fut.set_result(event)

        self.emit("generation_created", event)
        logger.debug(f"Started new generation {gen_id}")

    def _end_current_generation(self) -> None:
        """Signal end of current generation's output streams."""
        if self._current_generation_id is None:
            return
        self._audio_output_queue.put_nowait(None)
        self._text_output_queue.put_nowait(None)
        self._current_generation_id = None

    # -- Internal: WebSocket connection --

    async def _run(self) -> None:
        """Main connection loop — connect to Moshi, handshake, stream."""
        try:
            # Build WebSocket URL
            params: dict[str, str] = {}
            if self._opts.text_prompt:
                params["text_prompt"] = self._opts.text_prompt
            else:
                params["text_prompt"] = ""
            if self._opts.voice:
                params["voice_prompt"] = self._opts.voice
            else:
                params["voice_prompt"] = ""
            params["text_temperature"] = str(self._opts.text_temperature)
            params["text_topk"] = str(self._opts.text_topk)
            params["audio_temperature"] = str(self._opts.audio_temperature)
            params["audio_topk"] = str(self._opts.audio_topk)
            if self._opts.seed >= 0:
                params["seed"] = str(self._opts.seed)

            url = f"{self._opts.server_url}/api/chat"
            if params:
                url += "?" + urlencode(params)

            logger.info(f"Connecting to Moshi server at {self._opts.server_url}")

            async with aiohttp.ClientSession() as http:
                async with http.ws_connect(
                    url,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as ws:
                    self._ws = ws

                    # Initialize Opus codecs
                    self._opus_writer = sphn.OpusStreamWriter(MOSHI_SAMPLE_RATE)
                    self._opus_reader = sphn.OpusStreamReader(MOSHI_SAMPLE_RATE)

                    # Wait for handshake (0x00)
                    logger.info("Waiting for Moshi handshake...")
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        if len(msg.data) > 0 and msg.data[0] == 0x00:
                            logger.info("Moshi handshake received — session active")
                            self._connected.set()
                        else:
                            raise ConnectionError(
                                f"Unexpected handshake data: {msg.data[:10]!r}"
                            )
                    else:
                        raise ConnectionError(
                            f"Expected binary handshake, got {msg.type}"
                        )

                    # Emit initial generation
                    if self._pending_generation_fut:
                        self._start_new_generation(
                            self._pending_generation_fut, user_initiated=True
                        )
                        self._pending_generation_fut = None
                    else:
                        self._start_new_generation(user_initiated=False)

                    # Run bidirectional streaming
                    send_task = asyncio.create_task(self._send_loop(ws))
                    recv_task = asyncio.create_task(self._recv_loop(ws))

                    try:
                        done, pending = await asyncio.wait(
                            [send_task, recv_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in pending:
                            task.cancel()
                        for task in done:
                            exc = task.exception()
                            if exc is not None:
                                raise exc
                    except asyncio.CancelledError:
                        send_task.cancel()
                        recv_task.cancel()
                        raise

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Moshi connection error: {e}")
            self.emit(
                "error",
                RealtimeModelError(
                    type="realtime_model_error",
                    timestamp=time.time(),
                    label=self._realtime_model.label,
                    error=e,
                    recoverable=False,
                ),
            )
        finally:
            self._end_current_generation()

    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Receive audio and text from Moshi, push to output queues."""
        async for msg in ws:
            if self._closed:
                break

            if msg.type == aiohttp.WSMsgType.BINARY:
                data = msg.data
                if len(data) == 0:
                    continue

                kind = data[0]
                payload = data[1:]

                if kind == 0x01 and len(payload) > 0:
                    # Opus audio — decode to PCM
                    assert self._opus_reader is not None
                    self._opus_reader.append_bytes(payload)
                    while True:
                        pcm = self._opus_reader.read_pcm()
                        if pcm.shape[-1] == 0:
                            break
                        # Convert float32 numpy -> int16 bytes for AudioFrame
                        pcm_int16 = (
                            np.clip(pcm * 32767, -32768, 32767)
                            .astype(np.int16)
                        )
                        frame = rtc.AudioFrame(
                            data=pcm_int16.tobytes(),
                            sample_rate=MOSHI_SAMPLE_RATE,
                            num_channels=1,
                            samples_per_channel=len(pcm_int16),
                        )
                        try:
                            self._audio_output_queue.put_nowait(frame)
                        except asyncio.QueueFull:
                            pass

                elif kind == 0x02 and len(payload) > 0:
                    # Text token
                    text = payload.decode("utf-8")
                    try:
                        self._text_output_queue.put_nowait(text)
                    except asyncio.QueueFull:
                        pass

            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
                break

            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            ):
                break

    async def _send_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Read PCM AudioFrames from LiveKit, encode to Opus, send to Moshi."""
        assert self._opus_writer is not None

        while not self._closed:
            try:
                frame = await asyncio.wait_for(
                    self._audio_input_queue.get(),
                    timeout=0.1,
                )
            except asyncio.TimeoutError:
                continue

            try:
                # Extract PCM from AudioFrame (int16 bytes)
                pcm_bytes = bytes(frame.data)
                pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                pcm_float = pcm_int16.astype(np.float32) / 32768.0

                # Resample to 24kHz if needed
                if frame.sample_rate != MOSHI_SAMPLE_RATE:
                    ratio = MOSHI_SAMPLE_RATE / frame.sample_rate
                    new_length = int(len(pcm_float) * ratio)
                    if new_length > 0:
                        pcm_float = np.interp(
                            np.linspace(0, len(pcm_float) - 1, new_length),
                            np.arange(len(pcm_float)),
                            pcm_float,
                        )

                # Downmix to mono if multi-channel
                if frame.num_channels > 1:
                    pcm_float = pcm_float.reshape(-1, frame.num_channels).mean(
                        axis=1
                    )

                # Encode to Opus and send
                self._opus_writer.append_pcm(pcm_float)
                encoded = self._opus_writer.read_bytes()
                if len(encoded) > 0:
                    await ws.send_bytes(b"\x01" + encoded)

            except Exception as e:
                logger.error(f"Error encoding/sending audio: {e}")
