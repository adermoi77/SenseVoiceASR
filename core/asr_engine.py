"""
ASR 流式识别引擎模块

基于 FunASR 官方 SenseVoice 模型实现真正的流式语音识别
支持增量中间结果与最终结果的实时输出
"""

import logging
from dataclasses import dataclass
from typing import Optional, Generator, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ASRConfig:
    """ASR 配置数据类"""
    
    model_name: str = "sensevoice_small"
    device: str = "cpu"
    language: str = "auto"
    use_vad: bool = True
    use_itn: bool = True
    sample_rate: int = 16000
    
    # 高级参数
    max_single_segment_time: int = 30000  # 最大单段时长 (ms)
    vad_max_segment_length: int = 30000   # VAD 最大分段长度 (ms)
    

class StreamASREngine:
    """
    流式 ASR 识别引擎
    
    基于 FunASR 官方 SenseVoice 模型，实现真正的流式识别。
    支持音频分块输入、增量识别结果输出。
    
    Attributes:
        config: ASR 配置对象
        model: 加载的 ASR 模型实例
        is_initialized: 模型是否已初始化
    """
    
    def __init__(self, config: ASRConfig) -> None:
        """
        初始化 ASR 引擎
        
        Args:
            config: ASR 配置对象
        """
        self.config = config
        self.model: Optional[Any] = None
        self._cache: Optional[Any] = None
        self.is_initialized: bool = False
        
    def initialize(self) -> None:
        """
        初始化模型（延迟加载）
        
        从 ModelScope 自动下载并加载 SenseVoice 模型
        """
        if self.is_initialized:
            logger.info("模型已初始化，跳过")
            return
            
        try:
            logger.info(f"正在加载 FunASR 模型：{self.config.model_name}")
            logger.info(f"设备：{self.config.device}")
            
            from funasr import AutoModel
            
            # 使用 FunASR 官方 AutoModel 接口加载流式模型
            self.model = AutoModel(
                model=self.config.model_name,
                device=self.config.device,
                ncpu=4 if self.config.device == "cpu" else 1,
                disable_update=True,  # 不自动检查更新
            )
            
            self.is_initialized = True
            logger.info("模型加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败：{e}")
            raise RuntimeError(f"FunASR 模型初始化失败：{e}")
    
    def generate_stream(
        self, 
        audio_chunk: np.ndarray,
        is_final: bool = False
    ) -> Generator[dict, None, None]:
        """
        流式识别生成器
        
        接收音频数据块，实时返回识别结果
        
        Args:
            audio_chunk: 音频数据块 (numpy array, float32, 16kHz)
            is_final: 是否为最后一个音频块
            
        Yields:
            dict: 识别结果，包含：
                - text: 识别文本
                - is_final: 是否为最终结果
                - confidence: 置信度 (可选)
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # 归一化音频到 float32 [-1, 1]
            if audio_chunk.dtype == np.int16:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            elif audio_chunk.dtype == np.float32:
                audio_float = audio_chunk
            else:
                audio_float = audio_chunk.astype(np.float32)
            
            # 调用 FunASR 流式推理接口
            result = self.model.generate(
                input=audio_float,
                cache=self._cache,
                is_final=is_final,
                language=self.config.language,
                use_itn=self.config.use_itn,
            )
            
            # 解析结果
            if result and len(result) > 0:
                for item in result:
                    yield {
                        "text": item.get("text", ""),
                        "is_final": is_final,
                        "confidence": item.get("confidence", 1.0),
                    }
                    
        except Exception as e:
            logger.error(f"流式识别错误：{e}")
            yield {
                "text": "",
                "is_final": False,
                "error": str(e),
            }
    
    def reset(self) -> None:
        """重置识别状态（用于新会话）"""
        self._cache = None
        logger.debug("ASR 引擎状态已重置")
        
    def process_file(self, audio_path: str) -> str:
        """
        处理完整音频文件（测试用）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            str: 完整识别结果
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            import torchaudio
            
            waveform, sr = torchaudio.load(audio_path)
            
            # 重采样到 16kHz
            if sr != self.config.sample_rate:
                from torchaudio.transforms import Resample
                resampler = Resample(sr, self.config.sample_rate)
                waveform = resampler(waveform)
            
            # 转 mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 转 numpy
            audio_np = waveform.squeeze().numpy()
            
            # 整段识别
            result = self.model.generate(
                input=audio_np,
                language=self.config.language,
                use_itn=self.config.use_itn,
            )
            
            if result and len(result) > 0:
                return result[0].get("text", "")
            return ""
            
        except Exception as e:
            logger.error(f"文件识别错误：{e}")
            return f"[错误：{e}]"
