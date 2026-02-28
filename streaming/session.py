"""
Session state management for the prosodic pipeline.

InMemorySessionStore for MVP (single process).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import time


@dataclass
class SessionState:
    """
    Complete state for a streaming conversation session.
    """

    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    ssm_state: Optional[dict] = None
    predictor_state: Optional[Any] = None
    prosody_history: list = field(default_factory=list)
    last_predictions: Optional[dict] = None
    frames_processed: int = 0
    source: str = "unknown"
    vertical: Optional[str] = None
    api_key: Optional[str] = None


class SessionStore(ABC):
    """Abstract session state store."""

    @abstractmethod
    async def get(self, session_id: str) -> Optional[SessionState]:
        """Get session state. Returns None if not found."""
        ...

    @abstractmethod
    async def set(self, state: SessionState) -> None:
        """Save session state."""
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete session state."""
        ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        ...

    @abstractmethod
    async def active_count(self) -> int:
        """Number of active sessions."""
        ...


class InMemorySessionStore(SessionStore):
    """In-memory session store for single-process deployment."""

    def __init__(self, max_sessions: int = 10000):
        self._sessions: dict[str, SessionState] = {}
        self._max_sessions = max_sessions

    async def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    async def set(self, state: SessionState) -> None:
        state.updated_at = time.time()
        if len(self._sessions) >= self._max_sessions and state.session_id not in self._sessions:
            oldest_id = min(self._sessions, key=lambda k: self._sessions[k].updated_at)
            del self._sessions[oldest_id]
        self._sessions[state.session_id] = state

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def active_count(self) -> int:
        return len(self._sessions)
