# SenseVoice 实时流式 ASR 服务

基于 FunASR 官方 SenseVoice 模型的高性能流式语音识别服务，支持 WebSocket 全双工通信。

## 项目结构

```
SenseVoiceASR/
├─ core/                  # 核心模块
│   ├─ __init__.py       # 模块导出
│   ├─ asr_engine.py     # ASR 流式识别引擎
│   └─ ws_service.py     # WebSocket 服务
├─ example/              # 示例代码
│   ├─ test_server.py    # WebSocket 客户端测试
│   └─ test_engine.py    # ASR 引擎测试
├─ audio_example/        # 示例音频
│   └─ test.wav
├─ main.py               # 服务入口
├─ config.yaml           # 配置文件
└─ pyproject.toml        # 项目配置
```

## 环境要求

- Python >= 3.12
- uv (包管理工具)

## 快速开始

### 1. 安装依赖

```bash
cd /workspace
uv sync
```

### 2. 启动服务

```bash
# 方式 1: 使用 Python 直接运行
python main.py

# 方式 2: 使用 uv 运行
uv run python main.py
```

### 3. 测试连接

```bash
# 测试 WebSocket 连接
python example/test_server.py --connect-only

# 测试音频文件识别
python example/test_server.py --audio audio_example/test.wav

# 测试 ASR 引擎（单独）
python example/test_engine.py --load-only
```

## 配置说明

编辑 `config.yaml` 进行配置：

```yaml
server:
  host: "0.0.0.0"      # 监听地址
  port: 8765           # 监听端口
  max_connections: 100 # 最大连接数
  
asr:
  model_name: "sensevoice_small"  # 模型名称
  device: "cpu"                   # 运行设备：cpu/cuda
  language: "auto"                # 语言：auto/zh/en/ja/ko
  use_vad: true                   # 启用 VAD
  use_itn: true                   # 启用逆文本标准化
```

## WebSocket 协议

### 客户端发送格式

```json
// 发送音频数据
{
  "type": "audio",
  "data": "<base64_encoded_pcm>",
  "is_final": false
}

// 结束识别
{"type": "end"}

// 心跳
{"type": "ping"}

// 请求统计
{"type": "stats"}
```

### 服务端返回格式

```json
// 欢迎消息
{
  "type": "welcome",
  "session_id": "session_1_xxx",
  "message": "已连接到 SenseVoice ASR 服务",
  "config": {"sample_rate": 16000, "bit_depth": 16, "channels": 1}
}

// 中间结果
{
  "type": "partial",
  "text": "正在识别...",
  "session_id": "session_1_xxx",
  "confidence": 0.95
}

// 最终结果
{
  "type": "final",
  "text": "这是最终识别结果",
  "session_id": "session_1_xxx",
  "confidence": 0.98
}

// 错误消息
{
  "type": "error",
  "error": "错误描述",
  "session_id": "session_1_xxx"
}
```

## 音频格式要求

- 采样率：16000 Hz
- 位深：16 bit
- 声道：单声道 (Mono)
- 编码：PCM (未压缩)
- 传输：Base64 编码

## 特性

- ✅ 真正的流式识别（非伪流式）
- ✅ 增量中间结果输出
- ✅ 支持 VAD 语音活动检测
- ✅ 自动标点与逆文本标准化
- ✅ 多语言支持（中/英/日/韩）
- ✅ WebSocket 全双工通信
- ✅ 会话管理与性能统计
- ✅ 优雅关闭与异常处理
- ✅ 结构化日志输出

## 常见问题

### Q: 首次启动很慢？
A: 首次启动会自动从 ModelScope 下载模型，请耐心等待。后续启动会使用缓存。

### Q: 如何使用 GPU 加速？
A: 修改 `config.yaml` 中的 `device: cuda`，确保已安装 CUDA 版本的 PyTorch。

### Q: 识别准确率不高？
A: 请确保输入音频为 16kHz/16bit/单声道格式，可尝试调整 VAD 参数或使用更高质量的麦克风。

## License

MIT License
