import asyncio
import os
import time
import traceback

import pyaudio
from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-3.1-flash-live-preview"


def _make_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY nicht gesetzt")
    return genai.Client(http_options={"api_version": "v1beta"}, api_key=api_key)


def _make_config(system_prompt: str) -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        media_resolution="MEDIA_RESOLUTION_MEDIUM",
        system_instruction=system_prompt if system_prompt else None,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
            )
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=104857,
            sliding_window=types.SlidingWindow(target_tokens=52428),
        ),
    )


class GeminiVoiceSession:
    def __init__(self, system_prompt: str = "") -> None:
        self._system_prompt = system_prompt
        self._task: asyncio.Task | None = None
        self._pya: pyaudio.PyAudio | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name="gemini-voice"
        )
        print("[gemini_voice] Session gestartet")

    async def stop(self) -> None:
        task = self._task
        if task is None or task.done():
            self._task = None
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None
        print("[gemini_voice] Session gestoppt")

    async def _run(self) -> None:
        try:
            self._pya = pyaudio.PyAudio()
            client = _make_client()
            config = _make_config(self._system_prompt)

            audio_in_queue: asyncio.Queue[bytes] = asyncio.Queue()
            out_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=5)
            state = {"is_playing": False, "playback_end_time": 0.0}

            async with client.aio.live.connect(model=MODEL, config=config) as session:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._listen_audio(out_queue, state))
                    tg.create_task(self._send_realtime(session, out_queue))
                    tg.create_task(self._receive_audio(session, audio_in_queue))
                    tg.create_task(self._play_audio(audio_in_queue, out_queue, state))
                    # Keep running until cancelled
                    tg.create_task(asyncio.sleep(float("inf")))

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as eg:
            print(f"[gemini_voice] ExceptionGroup: {eg}")
            traceback.print_exception(eg)
        except Exception as exc:
            print(f"[gemini_voice] Fehler: {exc}")
        finally:
            if self._pya is not None:
                self._pya.terminate()
                self._pya = None

    async def _listen_audio(self, out_queue: asyncio.Queue, state: dict) -> None:
        mic_info = await asyncio.to_thread(self._pya.get_default_input_device_info)
        stream = await asyncio.to_thread(
            self._pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        try:
            while True:
                data = await asyncio.to_thread(
                    stream.read, CHUNK_SIZE, **{"exception_on_overflow": False}
                )
                cooldown = (time.monotonic() - state["playback_end_time"]) < 0.4
                if not state["is_playing"] and not cooldown:
                    await out_queue.put({"data": data, "mime_type": "audio/pcm"})
        finally:
            await asyncio.to_thread(stream.close)

    async def _send_realtime(self, session, out_queue: asyncio.Queue) -> None:
        while True:
            msg = await out_queue.get()
            if msg.get("mime_type") == "audio/pcm":
                await session.send_realtime_input(
                    audio=types.Blob(data=msg["data"], mime_type="audio/pcm")
                )

    async def _receive_audio(self, session, audio_in_queue: asyncio.Queue) -> None:
        while True:
            turn = session.receive()
            async for response in turn:
                if data := response.data:
                    audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="", flush=True)
            # Turn complete — drain stale audio to handle model interruption
            while not audio_in_queue.empty():
                audio_in_queue.get_nowait()

    async def _play_audio(
        self, audio_in_queue: asyncio.Queue, out_queue: asyncio.Queue, state: dict
    ) -> None:
        stream = await asyncio.to_thread(
            self._pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE * 4,
        )
        try:
            while True:
                bytestream = await audio_in_queue.get()
                # Coalesce all immediately available chunks to reduce gaps
                while True:
                    try:
                        bytestream += audio_in_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                state["is_playing"] = True
                await asyncio.to_thread(stream.write, bytestream)
                if audio_in_queue.empty():
                    state["is_playing"] = False
                    state["playback_end_time"] = time.monotonic()
                    # Drain mic data accumulated during playback
                    while True:
                        try:
                            out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
        finally:
            await asyncio.to_thread(stream.close)
