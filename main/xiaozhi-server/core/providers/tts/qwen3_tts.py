import os
import uuid
import time
import json
import base64
import queue
import asyncio
import traceback
import requests

from typing import Callable, Any
from config.logger import setup_logging
from core.utils.tts import MarkdownCleaner
from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import SentenceType, ContentType, InterfaceType

TAG = __name__
logger = setup_logging()


# ── Callback (duck-type compatible with QwenTtsRealtimeCallback) ──────────

class Qwen3TTSCallback:
    """WebSocket 事件回调 —— 将 Qwen3-TTS 音频 delta 编码为 Opus 帧入队

    由 QwenTtsRealtime 内部线程调用。provider 引用用于访问 opus_encoder 和音频队列。
    """

    def __init__(self, provider):
        self.provider = provider

    def on_open(self):
        logger.bind(tag=TAG).debug("Qwen3-TTS WebSocket 连接已建立")

    def on_event(self, event: dict):
        etype = event.get("type", "")
        if etype == "session.created":
            sid = event.get("session", {}).get("id", "unknown")
            logger.bind(tag=TAG).debug(f"会话创建: {sid}")
        elif etype == "session.updated":
            logger.bind(tag=TAG).debug("会话配置已应用")
        elif etype == "response.created":
            logger.bind(tag=TAG).debug("TTS 合成任务已启动")
        elif etype == "response.audio.delta":
            try:
                pcm_bytes = base64.b64decode(event["delta"])
                self.provider.handle_audio_delta(pcm_bytes)
            except Exception as e:
                logger.bind(tag=TAG).error(f"处理音频 delta 失败: {e}")
        elif etype == "response.audio.done":
            self.provider.handle_audio_done()
        elif etype == "error":
            logger.bind(tag=TAG).error(f"TTS 服务端错误: {event}")
        # 忽略不认识的事件类型

    def on_close(self, code, msg):
        logger.bind(tag=TAG).debug(f"WebSocket 关闭: code={code} msg={msg}")

    def on_error(self, error):
        logger.bind(tag=TAG).error(f"WebSocket 错误: {error}")


# ── Provider ───────────────────────────────────────────────────────────────

class TTSProvider(TTSProviderBase):
    """Qwen3-TTS WebSocket 实时流式 Provider（方案 1A）

    架构：Dual-Stream（参考 alibl_stream.py）
    - LLM token → tts_text_queue → tts_text_priority_thread
    - FIRST: start_session → WebSocket connect → session.update
    - TEXT:  text_to_speak → input_text_buffer.append
    - LAST:  finish_session → session.finish（保持 WS 连接）
    - 后台:  QwenTtsRealtime 内部线程 → callback.on_event → handle_audio_delta

    语音：Jada（上海-阿珍）——原生内置沪语（吴语）发音，零声音复刻。
    """

    TTS_PARAM_CONFIG = [
        ("ttsVolume", "volume", 0, 100, 50, int),
        ("ttsRate", "rate", 0.5, 2.0, 1.0, lambda v: round(float(v), 1)),
    ]

    # ── 生命周期 ──────────────────────────────────────────────────────

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        self.interface_type = InterfaceType.DUAL_STREAM
        self.report_on_last = True  # 累积上报模式（全程单一 WS 流）

        # 认证
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("Qwen3TTS 需要 api_key（阿里百炼 DashScope API Key）")

        # 模型 & 音色
        self.model = config.get("model", "qwen3-tts-flash-realtime")
        self.voice = config.get("voice", "Jada")  # 上海-阿珍
        self.mode = config.get("mode", "server_commit")  # 服务端自动判断合成时机
        self.language_type = config.get("language_type", "Chinese")

        # 音频参数
        self.audio_format = config.get("format", "pcm")
        self.sample_rate = int(config.get("sample_rate", 24000))

        rate = config.get("rate", "0.9")
        self.rate = float(rate) if rate else 0.9

        volume = config.get("volume", "50")
        self.volume = int(volume) if volume else 50

        # 应用百分比调整（Java 管理端下发）
        self._apply_percentage_params(config)

        # WebSocket 状态
        self.ws = None  # QwenTtsRealtime 实例
        self.callback = None  # Qwen3TTSCallback 实例
        self.last_active_time = None  # 上次活跃时间，用于 60s 连接复用

    # ── 文本处理线程（与 alibl_stream.py 同模式）─────────────────────

    def tts_text_priority_thread(self):
        """流式 TTS 文本处理线程

        消息生命周期：
        - FIRST → start_session (建立/复用 WebSocket，发送 session.update)
        - TEXT  → text_to_speak (input_text_buffer.append 增量追加)
        - FILE  → 音频文件直接入播放队列
        - LAST  → finish_session (session.finish，保持 WS 连接)
        """
        while not self.conn.stop_event.is_set():
            try:
                message = self.tts_text_queue.get(timeout=1)

                if self.conn.client_abort:
                    try:
                        logger.bind(tag=TAG).info("收到打断信息，终止 TTS 文本处理线程")
                        asyncio.run_coroutine_threadsafe(
                            self.finish_session(self.conn.sentence_id),
                            loop=self.conn.loop,
                        )
                        continue
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"取消 TTS 会话失败: {str(e)}")
                        continue

                # 过滤旧消息
                if message.sentence_id != self.conn.sentence_id:
                    continue

                logger.bind(tag=TAG).debug(
                    f"收到TTS任务｜{message.sentence_type.name} ｜ {message.content_type.name}"
                    f" | 会话ID: {message.sentence_id}"
                )

                if message.sentence_type == SentenceType.FIRST:
                    self.reset_stream_state()
                    try:
                        logger.bind(tag=TAG).debug("开始启动 TTS 会话...")
                        future = asyncio.run_coroutine_threadsafe(
                            self.start_session(self.conn.sentence_id),
                            loop=self.conn.loop,
                        )
                        future.result(timeout=self.tts_timeout)
                        self.before_stop_play_files.clear()
                        logger.bind(tag=TAG).debug("TTS 会话启动成功")
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"启动 TTS 会话失败: {str(e)}")
                        continue

                elif ContentType.TEXT == message.content_type:
                    if message.content_detail:
                        try:
                            logger.bind(tag=TAG).debug(
                                f"开始发送 TTS 文本: {message.content_detail}"
                            )
                            future = asyncio.run_coroutine_threadsafe(
                                self.text_to_speak(message.content_detail, None),
                                loop=self.conn.loop,
                            )
                            future.result(timeout=self.tts_timeout)
                        except Exception as e:
                            logger.bind(tag=TAG).error(f"发送 TTS 文本失败: {str(e)}")
                            continue

                elif ContentType.FILE == message.content_type:
                    logger.bind(tag=TAG).info(
                        f"添加音频文件到待播放列表: {message.content_file}"
                    )
                    if message.content_file and os.path.exists(message.content_file):
                        self._process_audio_file_stream(
                            message.content_file,
                            callback=lambda audio_data: self.handle_audio_file(
                                audio_data, message.content_detail
                            ),
                        )

                if message.sentence_type == SentenceType.LAST:
                    try:
                        logger.bind(tag=TAG).debug("开始结束 TTS 会话...")
                        future = asyncio.run_coroutine_threadsafe(
                            self.finish_session(self.conn.sentence_id),
                            loop=self.conn.loop,
                        )
                        future.result()
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"结束 TTS 会话失败: {str(e)}")
                        continue

            except queue.Empty:
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"处理 TTS 文本失败: {str(e)}, 类型: {type(e).__name__}, "
                    f"堆栈: {traceback.format_exc()}"
                )
                continue

    # ── 会话管理 ──────────────────────────────────────────────────────

    async def text_to_speak(self, text, _):
        """重写基类方法：通过 WebSocket 发送增量文本（非生成音频文件）

        ⚠️ Dual-Stream 模式下必须重写 —— 基类默认实现会尝试调用 HTTP API 生成文件。
        此处改为调用 QwenTtsRealtime.append_text() 将文本追加到服务端缓冲区。
        Qwen3-TTS 在 server_commit 模式下会自动判断合成时机并推送 response.audio.delta。
        """
        if self.ws is None:
            logger.bind(tag=TAG).warning("WebSocket 未连接，跳过文本发送")
            return

        filtered_text = MarkdownCleaner.clean_markdown(text)
        if filtered_text:
            # 使用滑动窗口匹配处理跨分片的替换词
            confirmed_texts, self._pending_prefix = self._match_stream_text(
                filtered_text
            )
            for txt in confirmed_texts:
                if txt:
                    self.ws.append_text(txt)
                    self.last_active_time = time.time()

    async def start_session(self, session_id):
        """建立 WebSocket 连接，发送 session.update

        60 秒内复用已有连接（参考 alibl_stream.py:73-81）。
        QwenTtsRealtime.connect() 可能是阻塞的，在线程池中运行。
        """
        logger.bind(tag=TAG).debug(f"开始会话～～{session_id}")
        try:
            current_time = time.time()
            if (
                self.ws
                and self.last_active_time
                and (current_time - self.last_active_time < 60)
            ):
                logger.bind(tag=TAG).debug("复用已有 WebSocket 连接")
                return

            # 关闭旧连接
            if self.ws:
                await self._close_ws()

            from dashscope.audio.qwen_tts import QwenTtsRealtime

            self.callback = Qwen3TTSCallback(self)
            self.ws = QwenTtsRealtime(
                model=self.model,
                api_key=self.api_key,
                callback=self.callback,
            )

            # connect() 内部建立 WebSocket 并启动后台接收线程
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.ws.connect)

            # 发送会话配置
            self.ws.update_session(
                voice=self.voice,
                mode=self.mode,
                sample_rate=self.conn.sample_rate,
                speech_rate=self.rate,
                volume=self.volume,
                audio_format=self.audio_format,
            )
            self.last_active_time = current_time
            logger.bind(tag=TAG).debug("会话启动请求已发送")

        except Exception as e:
            logger.bind(tag=TAG).error(f"启动会话失败: {str(e)}")
            await self._close_ws()
            raise

    async def finish_session(self, session_id):
        """结束当前句子的合成（不关闭 WebSocket 连接）

        Qwen3-TTS session.finish 结束服务端合成会话但保持 WS 连接。
        下一个句子通过 start_session() 在 60 秒内复用同一连接（不重新握手）。
        仅在 close() 中才真正断开 WebSocket。
        """
        logger.bind(tag=TAG).debug(f"关闭会话～～{session_id}")
        try:
            if self.ws:
                self.ws.finish()
                self.last_active_time = time.time()
        except Exception as e:
            logger.bind(tag=TAG).error(f"结束会话失败: {str(e)}")
            await self._close_ws()
            raise

    # ── 音频处理 ──────────────────────────────────────────────────────

    def handle_audio_delta(self, pcm_bytes: bytes):
        """将 PCM delta 编码为 Opus，放入播放队列

        ⚠️ 线程安全说明：
        此方法由 QwenTtsRealtime 内部接收线程调用（非 event loop 线程），
        tts_text_priority_thread 调用 text_to_speak 发送文本。
        - self.opus_encoder: 线程安全（独立 encoder buffer）
        - self.tts_audio_queue (queue.Queue): 线程安全
        - 播放预录音频文件时避免共享 opus_encoder（见 audio_to_opus_data_stream）
        """
        if pcm_bytes:
            self.opus_encoder.encode_pcm_to_opus_stream(
                pcm_bytes, False, callback=self.handle_opus
            )

    def handle_audio_done(self):
        """当前句子合成完成"""
        logger.bind(tag=TAG).debug("TTS 句子合成完成")

    # ── 资源清理 ──────────────────────────────────────────────────────

    async def _close_ws(self):
        """内部：安全关闭 WebSocket 连接"""
        if self.ws:
            try:
                self.ws.finish()
            except Exception:
                pass
            self.ws = None
        self.last_active_time = None

    async def close(self):
        """清理所有资源（WebSocket 连接 + 父类状态）"""
        await self._close_ws()
        await super().close()

    # ── 文件音频处理（独立编码器防并发）──────────────────────────────

    def audio_to_opus_data_stream(
        self, audio_file_path, callback: Callable[[Any], Any] = None
    ):
        """重写父类方法：使用独立的临时编码器处理音频文件，
        避免与 TTS 流式编码器并发冲突（参考 alibl_stream.py:403-419）。

        双流式 TTS 中，QwenTtsRealtime 内部线程使用 self.opus_encoder 编码
        实时音频 delta，同时 tts_text_priority_thread 处理音乐文件也使用
        self.opus_encoder。共享 encoder buffer 非线程安全，可能导致
        SILK resampler 断言失败。
        """
        from core.utils.util import audio_to_data_stream

        return audio_to_data_stream(
            audio_file_path,
            is_opus=True,
            callback=callback,
            sample_rate=self.conn.sample_rate,
            opus_encoder=None,  # 使用独立的临时编码器
        )

    # ── 同步模式兼容（测试用）─────────────────────────────────────────

    def to_tts(self, text: str) -> list:
        """非流式生成音频数据，用于测试场景（不依赖 ConnectionHandler）

        创建一个独立的 QwenTtsRealtime 会话，提交文本并收集所有音频 delta。
        返回 Opus 编码的音频数据列表。
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            audio_data = []

            async def _generate_audio():
                from dashscope.audio.qwen_tts import QwenTtsRealtime

                collected_pcm = []

                class _SyncCallback:
                    def on_open(self):
                        pass

                    def on_event(self, event):
                        etype = event.get("type", "")
                        if etype == "response.audio.delta":
                            try:
                                collected_pcm.append(
                                    base64.b64decode(event["delta"])
                                )
                            except Exception:
                                pass

                    def on_close(self, code, msg):
                        pass

                    def on_error(self, error):
                        pass

                cb = _SyncCallback()
                ws = QwenTtsRealtime(
                    model=self.model,
                    api_key=self.api_key,
                    callback=cb,
                )

                await loop.run_in_executor(None, ws.connect)
                ws.update_session(
                    voice=self.voice,
                    mode=self.mode,
                    sample_rate=self.conn.sample_rate,
                    speech_rate=self.rate,
                    volume=self.volume,
                    audio_format=self.audio_format,
                )

                # 文本预处理
                filtered = MarkdownCleaner.clean_markdown(text)
                if self._correct_words_pattern:
                    filtered = self._correct_words_pattern.sub(
                        lambda m: self.correct_words[m.group(0)], filtered
                    )

                ws.append_text(filtered)
                ws.finish()

                # 等待后台线程收集音频（简单延时）
                await asyncio.sleep(3)

                # 编码所有 PCM → Opus
                for pcm in collected_pcm:
                    self.opus_encoder.encode_pcm_to_opus_stream(
                        pcm,
                        end_of_stream=False,
                        callback=lambda opus: audio_data.append(opus),
                    )

            loop.run_until_complete(_generate_audio())
            loop.close()

            logger.bind(tag=TAG).debug(
                f"to_tts 生成 {len(audio_data)} 个 Opus 帧"
            )
            return audio_data

        except Exception as e:
            logger.bind(tag=TAG).error(f"to_tts 生成音频失败: {str(e)}")
            return []
