"""
SenseVoice ASR 核心模块

提供流式语音识别引擎与 WebSocket 服务封装
"""

from .asr_engine import StreamASREngine, ASRConfig
from .ws_service import WebSocketServer, SessionManager

__all__ = [
    "StreamASREngine",
    "ASRConfig", 
    "WebSocketServer",
    "SessionManager",
]

__version__ = "1.0.0"