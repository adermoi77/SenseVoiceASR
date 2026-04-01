"""
WebSocket 服务模块

提供基于 websockets 库的全双工通信服务
支持多客户端并发、心跳检测、会话管理
"""

import asyncio
import base64
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from .asr_engine import StreamASREngine, ASRConfig

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """会话信息数据类"""
    
    session_id: str
    websocket: WebSocketServerProtocol
    client_address: str
    connected_at: datetime = field(default_factory=datetime.now)
    audio_chunks_received: int = 0
    total_audio_duration_ms: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    is_processing: bool = False


class SessionManager:
    """
    会话管理器
    
    管理所有活跃的 WebSocket 连接会话
    提供会话创建、查询、删除和统计功能
    """
    
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_counter: int = 0
        
    def create_session(
        self, 
        websocket: WebSocketServerProtocol,
        client_address: str
    ) -> SessionInfo:
        """创建新会话"""
        self._session_counter += 1
        session_id = f"session_{self._session_counter}_{id(websocket)}"
        
        session = SessionInfo(
            session_id=session_id,
            websocket=websocket,
            client_address=client_address,
        )
        
        self._sessions[session_id] = session
        logger.info(f"新会话创建：{session_id} | 客户端：{client_address}")
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """获取会话信息"""
        return self._sessions.get(session_id)
    
    def remove_session(self, session_id: str) -> None:
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"会话关闭：{session_id}")
    
    def update_activity(self, session_id: str) -> None:
        """更新会话活动时间"""
        if session_id in self._sessions:
            self._sessions[session_id].last_activity = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        now = datetime.now()
        
        active_sessions = len(self._sessions)
        total_chunks = sum(s.audio_chunks_received for s in self._sessions.values())
        total_duration = sum(s.total_audio_duration_ms for s in self._sessions.values())
        
        return {
            "active_connections": active_sessions,
            "total_audio_chunks_received": total_chunks,
            "total_audio_duration_ms": round(total_duration, 2),
            "uptime_seconds": 0,
            "timestamp": now.isoformat(),
        }


class WebSocketServer:
    """
    WebSocket ASR 服务端
    
    提供流式语音识别的 WebSocket 接口
    支持 PCM 音频流的实时传输与识别结果推送
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        asr_config: ASRConfig,
        max_connections: int = 100,
        heartbeat_interval: int = 30,
    ) -> None:
        self.host = host
        self.port = port
        self.asr_config = asr_config
        self.max_connections = max_connections
        self.heartbeat_interval = heartbeat_interval
        
        self.session_manager = SessionManager()
        self.asr_engine: Optional[StreamASREngine] = None
        self._server: Optional[Any] = None
        self._is_running: bool = False
        self._start_time: Optional[datetime] = None
        self._total_requests: int = 0
        
    async def start(self) -> None:
        """启动 WebSocket 服务器"""
        try:
            logger.info("正在初始化 ASR 引擎...")
            self.asr_engine = StreamASREngine(self.asr_config)
            # 在线程池中同步执行初始化
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.asr_engine.initialize)
            
            self._server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                max_size=10 * 1024 * 1024,
                ping_interval=self.heartbeat_interval,
                ping_timeout=10,
            )
            
            self._is_running = True
            self._start_time = datetime.now()
            
            logger.info("=" * 50)
            logger.info("SenseVoice ASR 服务已启动")
            logger.info(f"监听地址：ws://{self.host}:{self.port}")
            logger.info(f"最大连接数：{self.max_connections}")
            logger.info(f"心跳间隔：{self.heartbeat_interval}秒")
            logger.info("=" * 50)
            
            await self._server.wait_closed()
            
        except Exception as e:
            logger.error(f"服务器启动失败：{e}")
            raise
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        """处理单个 WebSocket 连接"""
        client_address = websocket.remote_address
        session = None
        
        if len(self.session_manager._sessions) >= self.max_connections:
            await websocket.close(1013, "服务器已达最大连接数")
            logger.warning(f"拒绝连接：{client_address} (已达最大连接数)")
            return
        
        try:
            session = self.session_manager.create_session(websocket, str(client_address))
            
            await self._send_message(websocket, {
                "type": "welcome",
                "session_id": session.session_id,
                "message": "已连接到 SenseVoice ASR 服务",
                "config": {
                    "sample_rate": self.asr_config.sample_rate,
                    "bit_depth": 16,
                    "channels": 1,
                }
            })
            
            async for message in websocket:
                await self._process_message(session, message)
                
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"连接正常关闭：{session.session_id if session else 'unknown'} ({e.code})")
        except Exception as e:
            logger.error(f"连接异常：{session.session_id if session else 'unknown'} | {e}")
            if session and websocket.open:
                await self._send_error(websocket, str(e))
        finally:
            if session:
                self.session_manager.remove_session(session.session_id)
                if self.asr_engine:
                    self.asr_engine.reset()
    
    async def _process_message(self, session: SessionInfo, message: str) -> None:
        """处理单条 WebSocket 消息"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")
            
            self.session_manager.update_activity(session.session_id)
            
            if msg_type == "audio":
                await self._handle_audio(session, data)
            elif msg_type == "end":
                await self._handle_end(session)
            elif msg_type == "ping":
                await self._send_message(session.websocket, {"type": "pong"})
            elif msg_type == "stats":
                await self._handle_stats_request(session)
            else:
                logger.warning(f"未知消息类型：{msg_type}")
                await self._send_error(session.websocket, f"未知消息类型：{msg_type}", session.session_id)
                
        except json.JSONDecodeError:
            await self._handle_raw_audio(session, message)
        except Exception as e:
            logger.error(f"消息处理错误：{e}")
            await self._send_error(session.websocket, str(e), session.session_id)
    
    async def _handle_audio(self, session: SessionInfo, data: Dict[str, Any]) -> None:
        """处理音频数据消息"""
        audio_b64 = data.get("data", "")
        is_final = data.get("is_final", False)
        
        if not audio_b64:
            await self._send_error(session.websocket, "音频数据为空", session.session_id)
            return
        
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            
            session.audio_chunks_received += 1
            chunk_duration_ms = len(audio_np) / self.asr_config.sample_rate * 1000
            session.total_audio_duration_ms += chunk_duration_ms
            
            if self.asr_engine:
                for result in self.asr_engine.generate_stream(audio_np, is_final=is_final):
                    response_type = "final" if result.get("is_final") else "partial"
                    await self._send_message(session.websocket, {
                        "type": response_type,
                        "text": result.get("text", ""),
                        "session_id": session.session_id,
                        "confidence": result.get("confidence", 1.0),
                    })
                    
        except Exception as e:
            logger.error(f"音频处理错误：{e}")
            await self._send_error(session.websocket, f"音频处理失败：{e}", session.session_id)
    
    async def _handle_raw_audio(self, session: SessionInfo, raw_data: str) -> None:
        """处理原始音频数据"""
        try:
            audio_bytes = base64.b64decode(raw_data)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            
            session.audio_chunks_received += 1
            chunk_duration_ms = len(audio_np) / self.asr_config.sample_rate * 1000
            session.total_audio_duration_ms += chunk_duration_ms
            
            if self.asr_engine:
                for result in self.asr_engine.generate_stream(audio_np, is_final=False):
                    await self._send_message(session.websocket, {
                        "type": "partial",
                        "text": result.get("text", ""),
                        "session_id": session.session_id,
                    })
                    
        except Exception as e:
            logger.error(f"原始音频处理错误：{e}")
    
    async def _handle_end(self, session: SessionInfo) -> None:
        """处理结束消息"""
        logger.info(f"会话结束：{session.session_id}")
        
        if self.asr_engine:
            for result in self.asr_engine.generate_stream(np.array([], dtype=np.float32), is_final=True):
                await self._send_message(session.websocket, {
                    "type": "final",
                    "text": result.get("text", ""),
                    "session_id": session.session_id,
                })
        
        self._total_requests += 1
    
    async def _handle_stats_request(self, session: SessionInfo) -> None:
        """处理统计信息请求"""
        stats = self.session_manager.get_stats()
        if self._start_time:
            stats["uptime_seconds"] = (datetime.now() - self._start_time).total_seconds()
        stats["total_requests"] = self._total_requests
        
        await self._send_message(session.websocket, {
            "type": "stats",
            "data": stats,
        })
    
    async def _send_message(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]) -> None:
        """发送 JSON 消息"""
        try:
            await websocket.send(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"发送消息失败：{e}")
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error_msg: str, session_id: Optional[str] = None) -> None:
        """发送错误消息"""
        await self._send_message(websocket, {
            "type": "error",
            "error": error_msg,
            "session_id": session_id,
        })
    
    async def stop(self) -> None:
        """停止服务器"""
        self._is_running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("服务器已停止")
