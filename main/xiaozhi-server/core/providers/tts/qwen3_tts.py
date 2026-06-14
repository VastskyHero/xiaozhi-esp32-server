import os
import io
import math
import uuid
import traceback
import requests
import dashscope

from pydub import AudioSegment
from config.logger import setup_logging
from core.utils.tts import MarkdownCleaner
from core.providers.tts.base import TTSProviderBase

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    """Qwen3-TTS HTTP 非流式 Provider（方案 1B — 当前可用方案）

    使用阿里百炼 DashScope MultiModalConversation.call() 一次性提交文本，
    获取完整 WAV/PCM 音频后由基类自动处理 Opus 编码、重试、队列管理。

    音色：Jada（上海-阿珍）—— 原生内置沪语（吴语）发音，零声音复刻。

    架构：NON_STREAM（仅需实现 text_to_speak，参考 custom.py / doubao.py）

    注：WebSocket 实时流式（方案 1A）待 DashScope SDK 支持 QwenTtsRealtime
    或 Qwen3-TTS realtime 端点正式发布后再启用。当前 /api-ws/v1/realtime
    端点仅支持 qwen-omni-turbo-realtime 模型。
    """

    TTS_PARAM_CONFIG = [
        ("ttsVolume", "volume", 1, 100, 50, int),
        ("ttsRate", "rate", 0.5, 2.0, 1.0, lambda v: round(float(v), 1)),
    ]

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        # 认证
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("Qwen3TTS 需要 api_key（阿里百炼 DashScope API Key）")

        # 模型 & 音色
        self.model = config.get(
            "model", "qwen3-tts-flash-2025-11-27"
        )  # HTTP 非流式模型
        self.voice = config.get("voice", "Jada")  # 上海-阿珍，原生沪语
        self.language_type = config.get("language_type", "Chinese")

        # 音频参数
        self.audio_file_type = config.get("format", "wav")  # HTTP 模式输出 WAV
        self.sample_rate = int(config.get("sample_rate", 24000))

        rate = config.get("rate", "0.9")
        self.rate = float(rate) if rate else 0.9

        volume = config.get("volume", "50")
        self.volume = int(volume) if volume else 50

        # 应用百分比调整（Java 管理端下发）
        self._apply_percentage_params(config)

    async def text_to_speak(self, text: str, output_file: str):
        """调用 Qwen3-TTS HTTP API 生成语音

        基类自动处理：
        - Markdown 清洗 → 通过 MarkdownCleaner 在 to_tts_stream 中处理
        - 文本替换词（correct_words）→ 在基类 to_tts_stream 中处理
        - 重试（最多 5 次）→ 在基类 to_tts_stream 中处理
        - 音频文件 → Opus 编码 → 播放队列 → 在基类中处理
        """
        dashscope.api_key = self.api_key

        try:
            response = dashscope.MultiModalConversation.call(
                model=self.model,
                text=text,
                voice=self.voice,
                language_type=self.language_type,
                stream=False,
            )

            if response.status_code != 200:
                raise Exception(
                    f"TTS API 返回 {response.status_code}: "
                    f"{getattr(response, 'message', 'unknown error')}"
                )

            audio_url = response.output.audio.url
            audio_resp = requests.get(audio_url, timeout=30)

            if audio_resp.status_code != 200:
                raise Exception(f"音频下载失败: HTTP {audio_resp.status_code}")

            # ── 音量增益（服务端后处理）──────────────────────────────────
            # Qwen3-TTS HTTP API 不支持 volume 参数，用 pydub 在服务端放大。
            # volume=50 → 0dB（原始音量），volume=100 → +6dB
            audio_bytes = audio_resp.content
            if self.volume != 50:
                gain_db = 20 * math.log10(self.volume / 50.0)
                audio = AudioSegment.from_file(
                    io.BytesIO(audio_bytes),
                    format=self.audio_file_type,
                    parameters=["-nostdin"],
                )
                audio = audio.apply_gain(gain_db)
                buf = io.BytesIO()
                audio.export(buf, format=self.audio_file_type)
                audio_bytes = buf.getvalue()

            if output_file:
                with open(output_file, "wb") as f:
                    f.write(audio_bytes)
            else:
                return audio_bytes

        except Exception as e:
            logger.bind(tag=TAG).error(
                f"Qwen3TTS 语音生成失败: {text[:30]}... | "
                f"错误: {type(e).__name__}: {e}"
            )
            raise  # 抛出异常让基类的重试机制处理


# =============================================================================
# 未来方案 1A（WebSocket 实时流式）保留区
# =============================================================================
# 当以下条件之一满足时启用：
#   1. DashScope SDK 正式发布 QwenTtsRealtime 类
#   2. /api-ws/v1/realtime 端点支持 qwen3-tts-flash-realtime 模型
#   3. Qwen3-TTS 开源模型本地部署
#
# 实现参考（伪代码）：
#
#   from dashscope.audio.qwen_tts import QwenTtsRealtime
#
#   class TTSProvider(TTSProviderBase):
#       interface_type = InterfaceType.DUAL_STREAM
#       report_on_last = True
#
#       async def start_session(self, session_id):
#           self.ws = QwenTtsRealtime(model=..., api_key=..., callback=...)
#           self.ws.connect()
#           self.ws.update_session(voice="Jada", mode="server_commit", ...)
#
#       async def text_to_speak(self, text, _):
#           self.ws.append_text(text)  # 增量追加
#
#       async def finish_session(self, session_id):
#           self.ws.finish()
#
#   Provider 路径: core/providers/tts/qwen3_tts_stream.py
#   配置 type: qwen3_tts_stream
# =============================================================================
