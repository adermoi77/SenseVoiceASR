"""
SenseVoice ASR 服务主入口

提供美观的启动 Banner、配置加载、服务启动与优雅关闭
"""

import asyncio
import signal
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml

from core.asr_engine import ASRConfig, StreamASREngine
from core.ws_service import WebSocketServer

# 配置日志
def setup_logging(config: dict) -> logging.Logger:
    """配置日志系统"""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    log_file = log_config.get("file", "logs/asr_server.log")
    
    # 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置根日志
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ]
    )
    
    return logging.getLogger(__name__)


def print_banner(server_config: dict) -> None:
    """打印美观的启动 Banner"""
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║              SenseVoice 实时流式 ASR 服务                  ║
║              Based on FunASR & WebSocket                 ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  版本：1.0.0                                             ║
║  Python: {py_version:<48} ║
║  启动时间：{start_time:<45} ║
╠══════════════════════════════════════════════════════════╣
║  服务配置：                                               ║
║    监听地址：ws://{host:<39} ║
║    最大连接：  {max_conn:<43} ║
║    心跳间隔：  {heartbeat:<42}秒 ║
╠══════════════════════════════════════════════════════════╣
║  ASR 配置：                                                ║
║    模型：     {model:<44} ║
║    设备：     {device:<44} ║
║    语言：     {language:<44} ║
║    VAD:       {vad:<44} ║
║    ITN:       {itn:<44} ║
╚══════════════════════════════════════════════════════════╝
    """.format(
        py_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        host=f"{server_config['host']}:{server_config['port']}",
        max_conn=str(server_config['max_connections']),
        heartbeat=str(server_config['heartbeat_interval']),
        model=server_config['asr']['model_name'],
        device=server_config['asr']['device'],
        language=server_config['asr']['language'],
        vad="启用" if server_config['asr']['use_vad'] else "禁用",
        itn="启用" if server_config['asr']['use_itn'] else "禁用",
    )
    print(banner)


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ASRService:
    """ASR 服务管理类"""
    
    def __init__(self, config: dict) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.server: Optional[WebSocketServer] = None
        self._shutdown_event = asyncio.Event()
        
    async def start(self) -> None:
        """启动服务"""
        server_cfg = self.config["server"]
        asr_cfg = self.config["asr"]
        
        # 创建 ASR 配置
        asr_config = ASRConfig(
            model_name=asr_cfg["model_name"],
            device=asr_cfg["device"],
            language=asr_cfg["language"],
            use_vad=asr_cfg["use_vad"],
            use_itn=asr_cfg["use_itn"],
        )
        
        # 创建 WebSocket 服务器
        self.server = WebSocketServer(
            host=server_cfg["host"],
            port=server_cfg["port"],
            asr_config=asr_config,
            max_connections=server_cfg["max_connections"],
            heartbeat_interval=server_cfg["heartbeat_interval"],
        )
        
        # 启动服务
        await self.server.start()
        
    async def shutdown(self) -> None:
        """优雅关闭服务"""
        self.logger.info("正在关闭服务...")
        if self.server:
            await self.server.stop()
        self._shutdown_event.set()
        self.logger.info("服务已关闭")
        
    def wait_for_shutdown(self) -> asyncio.Event:
        """获取关闭事件"""
        return self._shutdown_event


async def main_async() -> None:
    """异步主函数"""
    try:
        # 加载配置
        config = load_config()
        
        # 设置日志
        logger = setup_logging(config)
        
        # 打印 Banner
        print_banner(config)
        
        # 创建服务
        service = ASRService(config)
        
        # 设置信号处理
        loop = asyncio.get_running_loop()
        
        def signal_handler():
            logger.info("收到退出信号")
            asyncio.create_task(service.shutdown())
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
        
        # 启动服务
        await service.start()
        
        # 等待关闭
        await service.wait_for_shutdown().wait()
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"服务异常：{e}")
        raise


def main() -> None:
    """同步主入口"""
    try:
        asyncio.run(main_async())
    except Exception as e:
        print(f"服务启动失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
