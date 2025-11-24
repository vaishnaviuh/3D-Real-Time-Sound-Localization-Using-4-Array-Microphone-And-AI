"""
State management for audio processing and tracking.
Replaces global state variables with a centralized state manager.
"""
from typing import Optional
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class ProcessingState:
    """Manages processing state for audio localization."""
    active: bool = False
    task: Optional[asyncio.Task] = None
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=2))
    smoothed_direction: Optional[np.ndarray] = None
    smoothed_confidence: float = 0.0
    smoothed_position: Optional[np.ndarray] = None
    
    def reset_smoothing(self):
        """Reset smoothing state."""
        self.smoothed_direction = None
        self.smoothed_confidence = 0.0
        self.smoothed_position = None
    
    def cleanup(self):
        """Clean up resources."""
        self.active = False
        if self.task and not self.task.done():
            self.task.cancel()
        self.executor.shutdown(wait=False)
        self.reset_smoothing()


@dataclass
class ConnectionState:
    """Manages WebSocket connection state."""
    active_connections: set = field(default_factory=set)
    
    def add_connection(self, websocket):
        """Add a WebSocket connection."""
        self.active_connections.add(websocket)
    
    def remove_connection(self, websocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
    
    def get_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)
    
    def has_connections(self) -> bool:
        """Check if there are any active connections."""
        return len(self.active_connections) > 0


class StateManager:
    """Centralized state manager for the application."""
    
    def __init__(self):
        self.processing = ProcessingState()
        self.connections = ConnectionState()
    
    def cleanup_all(self):
        """Clean up all state."""
        self.processing.cleanup()
        self.connections.active_connections.clear()

