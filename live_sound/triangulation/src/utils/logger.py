"""
Structured logging utilities for the sound localization system.
"""
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class StructuredLogger:
    """Structured logger that outputs JSON-formatted logs."""
    
    def __init__(self, name: str, log_file: Optional[Path] = None, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler with JSON formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _log_event(self, level: int, event_type: str, message: str, **kwargs):
        """Internal method to log structured events."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "message": message,
            **kwargs
        }
        self.logger.log(level, json.dumps(log_data))
    
    def info(self, message: str, event_type: str = "info", **kwargs):
        """Log info level message."""
        self._log_event(logging.INFO, event_type, message, **kwargs)
    
    def debug(self, message: str, event_type: str = "debug", **kwargs):
        """Log debug level message."""
        self._log_event(logging.DEBUG, event_type, message, **kwargs)
    
    def warning(self, message: str, event_type: str = "warning", **kwargs):
        """Log warning level message."""
        self._log_event(logging.WARNING, event_type, message, **kwargs)
    
    def error(self, message: str, event_type: str = "error", **kwargs):
        """Log error level message."""
        self._log_event(logging.ERROR, event_type, message, **kwargs)
    
    def exception(self, message: str, event_type: str = "exception", **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message)
        self._log_event(logging.ERROR, event_type, message, **kwargs)
    
    def log_audio_event(self, event: str, array_name: Optional[str] = None, **kwargs):
        """Log audio processing event."""
        self.info(
            f"Audio event: {event}",
            event_type="audio_processing",
            array_name=array_name,
            **kwargs
        )
    
    def log_doa_result(self, azimuth: float, elevation: float, confidence: float, **kwargs):
        """Log DOA estimation result."""
        self.debug(
            f"DOA: az={azimuth:.1f}°, el={elevation:.1f}°, conf={confidence:.3f}",
            event_type="doa_estimation",
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            confidence=confidence,
            **kwargs
        )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with additional context."""
        self.error(
            f"Error: {str(error)}",
            event_type="error",
            error_type=type(error).__name__,
            context=context
        )


# Global logger instance
_logger_instance: Optional[StructuredLogger] = None


def get_logger(name: str = "sound_localization") -> StructuredLogger:
    """Get or create the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = StructuredLogger(name)
    return _logger_instance


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO):
    """Setup global logging configuration."""
    global _logger_instance
    _logger_instance = StructuredLogger("sound_localization", log_file, level)
    return _logger_instance

