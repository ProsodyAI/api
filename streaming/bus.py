"""
AudioBus abstraction for audio transport.

WebSocketAudioBus for MVP (in-process, no external deps).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional
import asyncio


@dataclass
class AudioFrame:
    """Single frame of audio data flowing through the bus."""

    session_id: str
    pcm_data: bytes           # Raw PCM int16, mono, 16kHz
    sample_rate: int = 16000
    channels: int = 1
    timestamp_ms: float = 0.0
    sequence: int = 0         # Frame sequence number
    source: str = "unknown"   # "websocket", "livekit", "twilio"


class AudioBus(ABC):
    """
    Abstract audio transport layer.
    """

    @abstractmethod
    async def publish(self, frame: AudioFrame) -> None:
        """Publish an audio frame to the bus."""
        ...

    @abstractmethod
    async def subscribe(self, session_id: str) -> AsyncIterator[AudioFrame]:
        """Subscribe to audio frames for a session."""
        ...

    @abstractmethod
    async def close_session(self, session_id: str) -> None:
        """Signal end of session (no more frames)."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the bus."""
        ...


class WebSocketAudioBus(AudioBus):
    """
    In-process audio bus using asyncio.Queue per session.
    """

    def __init__(self, max_queue_size: int = 1000):
        self._queues: dict[str, asyncio.Queue[Optional[AudioFrame]]] = {}
        self._max_queue_size = max_queue_size

    def _get_queue(self, session_id: str) -> asyncio.Queue:
        if session_id not in self._queues:
            self._queues[session_id] = asyncio.Queue(maxsize=self._max_queue_size)
        return self._queues[session_id]

    async def publish(self, frame: AudioFrame) -> None:
        queue = self._get_queue(frame.session_id)
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(frame)

    async def subscribe(self, session_id: str) -> AsyncIterator[AudioFrame]:
        queue = self._get_queue(session_id)
        while True:
            frame = await queue.get()
            if frame is None:
                break
            yield frame

    async def close_session(self, session_id: str) -> None:
        if session_id in self._queues:
            await self._queues[session_id].put(None)

    async def shutdown(self) -> None:
        for session_id in list(self._queues.keys()):
            await self.close_session(session_id)
        self._queues.clear()
