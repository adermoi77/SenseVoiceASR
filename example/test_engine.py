"""
ASR 引擎测试脚本

用于单独测试 FunASR 模型加载和识别功能
"""

import logging
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.asr_engine import ASRConfig, StreamASREngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


def test_model_loading() -> None:
    """测试模型加载"""
    logger.info("=" * 50)
    logger.info("测试 1: 模型加载")
    logger.info("=" * 50)
    
    config = ASRConfig(
        model_name="sensevoice_small",
        device="cpu",
        language="auto",
        use_vad=True,
        use_itn=True,
    )
    
    engine = StreamASREngine(config)
    
    try:
        logger.info("开始加载模型...")
        engine.initialize()
        return True
    except Exception as e:
        logger.error(f"模型加载失败：{e}")
        return False


def test_file_recognition(audio_path: str) -> None:
    """测试文件识别"""
    logger.info("=" * 50)
    logger.info(f"测试 2: 文件识别 - {audio_path}")
    logger.info("=" * 50)
    
    config = ASRConfig(
        model_name="sensevoice_small",
        device="cpu",
        language="zh",  # 中文
        use_itn=True,
    )
    
    engine = StreamASREngine(config)
    
    try:
        result = engine.process_file(audio_path)
        logger.info(f"识别结果：{result}")
    except Exception as e:
        logger.error(f"文件识别失败：{e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASR 引擎测试工具")
    parser.add_argument("--audio", type=str, help="测试音频文件路径")
    parser.add_argument("--load-only", action="store_true", help="仅测试模型加载")
    
    args = parser.parse_args()
    
    # 测试模型加载
    if not test_model_loading():
        logger.error("模型加载失败，无法继续测试")
        return
    
    if args.load_only:
        logger.info("仅执行了模型加载测试")
        return
    
    # 测试文件识别
    if args.audio:
        audio_path = Path(args.audio)
        if audio_path.exists():
            test_file_recognition(str(audio_path))
        else:
            logger.error(f"音频文件不存在：{audio_path}")
    else:
        # 使用示例音频
        example_audio = Path(__file__).parent.parent / "audio_example" / "test.wav"
        if example_audio.exists():
            test_file_recognition(str(example_audio))
        else:
            logger.info("未找到示例音频文件")
            logger.info("使用方法：python test_engine.py --audio /path/to/audio.wav")


if __name__ == "__main__":
    main()
