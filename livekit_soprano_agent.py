"""LiveKit Agent: Faster Whisper STT → Gemini Flash → Soprano TTS"""

import os
import re
import numpy as np
from typing import AsyncIterable, Optional, List

from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, ModelSettings, stt
from livekit.agents import cli
from livekit.agents.worker import WorkerOptions
from livekit.agents.job import JobExecutorType
from livekit.plugins import google, silero
from google.genai.types import Modality

from faster_whisper import WhisperModel
from soprano import SopranoTTS


class FastPipelineAgent(Agent):
    def __init__(self, *, whisper_model: WhisperModel, soprano: SopranoTTS):
        super().__init__()
        self._whisper = whisper_model
        self._soprano = soprano
        self._sent_re = re.compile(r"(.+?[.!?]\s+|.+?\n+)", re.DOTALL)

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        async def _run() -> AsyncIterable[stt.SpeechEvent]:
            audio_data = []
            sample_rate = 16000
            
            async for frame in audio:
                samples_i16 = np.frombuffer(frame.data, dtype=np.int16)
                samples_f32 = samples_i16.astype(np.float32) / 32768.0
                audio_data.append(samples_f32)
                sample_rate = frame.sample_rate
            
            if not audio_data:
                yield stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, alternatives=[])
                return
            
            full_audio = np.concatenate(audio_data)
            segments, _ = self._whisper.transcribe(full_audio, beam_size=1, language="en")
            text = " ".join([seg.text for seg in segments]).strip()
            
            if text:
                yield stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text=text)],
                )
            
            yield stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, alternatives=[])

        return _run()

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:
        buffer = ""
        sample_rate = 32000
        channels = 1
        frame_ms = 20
        spf = int(sample_rate * frame_ms / 1000)

        async def _emit_audio_from_pcm(pcm_f32: np.ndarray):
            pcm_i16 = np.clip(pcm_f32, -1.0, 1.0)
            pcm_i16 = (pcm_i16 * 32767.0).astype(np.int16)
            idx = 0
            n = pcm_i16.shape[0]
            while idx < n:
                chunk = pcm_i16[idx : idx + spf]
                if chunk.shape[0] < spf:
                    pad = np.zeros((spf - chunk.shape[0],), dtype=np.int16)
                    chunk = np.concatenate([chunk, pad], axis=0)
                idx += spf
                yield rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=sample_rate,
                    num_channels=channels,
                    samples_per_channel=spf,
                )

        async def _speak_sentence(sentence: str):
            stream = self._soprano.infer_stream(sentence, chunk_size=1)
            for chunk in stream:
                if isinstance(chunk, np.ndarray):
                    pcm = chunk.astype(np.float32)
                else:
                    pcm = np.asarray(chunk, dtype=np.float32)
                async for frame in _emit_audio_from_pcm(pcm):
                    yield frame

        async for delta in text:
            buffer += delta
            out_sentences: List[str] = []
            while True:
                m = self._sent_re.match(buffer)
                if not m:
                    break
                s = m.group(1)
                out_sentences.append(s)
                buffer = buffer[len(s):]

            for s in out_sentences:
                async for frame in _speak_sentence(s):
                    yield frame

        if buffer.strip():
            async for frame in _speak_sentence(buffer):
                yield frame


def prewarm(proc: agents.JobProcess):
    # Silero VAD
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.05,
        min_silence_duration=0.35,
        force_cpu=True,
    )

    # Faster Whisper - tiny for speed
    print("Loading Faster Whisper...")
    proc.userdata["whisper"] = WhisperModel(
        "tiny",
        device="cuda",
        compute_type="float16",
    )
    print("✅ Whisper loaded")

    # Soprano TTS
    print("Loading Soprano TTS...")
    proc.userdata["soprano"] = SopranoTTS(device="cuda")
    print("✅ Soprano loaded")


async def entrypoint(ctx: agents.JobContext):
    vad = ctx.proc.userdata["vad"]
    whisper_model = ctx.proc.userdata["whisper"]
    soprano = ctx.proc.userdata["soprano"]

    agent = FastPipelineAgent(whisper_model=whisper_model, soprano=soprano)

    session = AgentSession(
        turn_detection="vad",
        vad=vad,
        llm=google.realtime.RealtimeModel(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            modalities=[Modality.TEXT],
            instructions=os.getenv(
                "SYSTEM_PROMPT",
                "You are a concise, helpful voice assistant. Keep replies short.",
            ),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
        ),
        preemptive_generation=True,
    )

    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # FIX 1: Give model downloads/init 5 minutes instead of 10s
            initialize_process_timeout=300.0,
            # FIX 2: Don't spawn multiple warm processes (they all download)
            num_idle_processes=0,
            # FIX 3: Use threads instead of processes (avoids Colab issues)
            job_executor_type=JobExecutorType.THREAD,
        )
    )
