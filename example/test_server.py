"""
测试服务器脚本

提供简单的 WebSocket 客户端测试功能
"""

import asyncio
import base64
import json
import wave
import logging
from pathlib import Path

import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_connection(server_url: str = "ws://localhost:8765") -> None:
    """测试连接到 ASR 服务"""
    try:
        async with websockets.connect(server_url) as ws:
            # 接收欢迎消息
            welcome = await ws.recv()
            logger.info(f"连接成功：{welcome}")
            
            # 发送 ping
            await ws.send(json.dumps({"type": "ping"}))
            pong = await ws.recv()
            logger.info(f"Ping 响应：{pong}")
            
            # 请求统计信息
            await ws.send(json.dumps({"type": "stats"}))
            stats = await ws.recv()
            logger.info(f"统计信息：{stats}")
            
    except Exception as e:
        logger.error(f"连接测试失败：{e}")


async def test_audio_file(
    server_url: str,
    audio_path: str,
    chunk_size: int = 3200  # 200ms @ 16kHz
) -> None:
    """测试音频文件识别"""
    try:
        async with websockets.connect(server_url) as ws:
            # 接收欢迎消息
            welcome = await ws.recv()
            logger.info(f"连接成功：{json.loads(welcome)}")
            
            # 读取 WAV 文件
            with wave.open(str(audio_path), "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                
                logger.info(f"音频信息：采样率={sample_rate}Hz, 声道={n_channels}, 位深={sample_width*8}bit")
                
                if sample_rate != 16000 or n_channels != 1 or sample_width != 2:
                    logger.warning("警告：音频格式不是 16kHz/16bit/单声道，可能需要转换")
                
                # 分块发送音频
                total_chunks = 0
                while True:
                    chunk = wf.readframes(chunk_size)
                    if not chunk:
                        break
                    
                    # 编码为 Base64
                    audio_b64 = base64.b64encode(chunk).decode("utf-8")
                    
                    # 发送音频数据
                    is_final = len(chunk) < chunk_size
                    await ws.send(json.dumps({
                        "type": "audio",
                        "data": audio_b64,
                        "is_final": is_final,
                    }))
                    total_chunks += 1
                    
                    # 接收识别结果
                    try:
                        while True:
                            result = await asyncio.wait_for(ws.recv(), timeout=0.1)
                            msg = json.loads(result)
                            if msg["type"] in ["partial", "final"]:
                                logger.info(f"识别结果 [{msg['type']}]: {msg.get('text', '')}")
                            else:
                                break
                    except asyncio.TimeoutError:
                        pass
                
                logger.info(f"已发送 {total_chunks} 个音频块")
                
                # 发送结束信号
                await ws.send(json.dumps({"type": "end"}))
                
                # 接收最终结果
                try:
                    final_result = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    logger.info(f"最终结果：{final_result}")
                except asyncio.TimeoutError:
                    logger.info("未收到最终结果")
                    
    except Exception as e:
        logger.error(f"音频测试失败：{e}")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASR 服务测试工具")
    parser.add_argument("--server", default="ws://localhost:8765", help="WebSocket 服务器地址")
    parser.add_argument("--audio", type=str, help="测试音频文件路径")
    parser.add_argument("--connect-only", action="store_true", help="仅测试连接")
    
    args = parser.parse_args()
    
    if args.connect_only:
        await test_connection(args.server)
    elif args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            logger.error(f"音频文件不存在：{audio_path}")
            return
        await test_audio_file(args.server, audio_path)
    else:
        # 默认使用示例音频
        example_audio = Path(__file__).parent.parent / "audio_example" / "test.wav"
        if example_audio.exists():
            await test_audio_file(args.server, example_audio)
        else:
            logger.info("未找到示例音频，执行连接测试...")
            await test_connection(args.server)


if __name__ == "__main__":
    asyncio.run(main())
