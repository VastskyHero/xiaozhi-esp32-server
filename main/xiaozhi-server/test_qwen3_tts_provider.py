#!/usr/bin/env python3
"""Qwen3TTS Provider 单元验证脚本（Step 1A.4）

在无需 ESP32 硬件的情况下验证 Provider 核心链路：
- start_session → append_text → handle_audio_delta → audio queue

用法：
    set DASHSCOPE_API_KEY=你的KEY
    python test_qwen3_tts_provider.py

前置条件：
    - Step 1A.2 WebSocket 探路已通过
    - qwen3_tts.py 已放入 core/providers/tts/
"""
import os
import sys
import time
import asyncio
import threading

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "你的APIKEY")

# 简单的 Mock 连接对象，模拟 ConnectionHandler
class MockConn:
    def __init__(self):
        self.sample_rate = 24000
        self.stop_event = threading.Event()
        self.client_abort = False
        self.sentence_id = "test-session-001"
        self.loop = asyncio.get_event_loop()
        self.audio_format = "pcm"
        self.headers = {}
        self.max_output_size = 0


async def test_provider():
    """验证 Provider 核心链路"""
    print("=" * 60)
    print("  Qwen3TTS Provider 单元验证（Step 1A.4）")
    print("=" * 60)

    if API_KEY == "你的APIKEY":
        print("❌ 请先设置 DASHSCOPE_API_KEY 环境变量")
        sys.exit(1)

    from core.providers.tts.qwen3_tts import TTSProvider
    from core.providers.tts.dto.dto import TTSMessageDTO, SentenceType, ContentType

    config = {
        "type": "qwen3_tts",
        "api_key": API_KEY,
        "model": "qwen3-tts-flash-realtime",
        "voice": "Jada",
        "mode": "server_commit",
        "rate": 0.9,
        "volume": 50,
        "format": "pcm",
        "sample_rate": 24000,
        "output_dir": "tmp/",
    }

    conn = MockConn()

    print("[1/4] 创建 Provider 实例 ...")
    try:
        provider = TTSProvider(config, delete_audio_file=False)
        provider.conn = conn
        print("   ✅ Provider 实例化成功")
    except Exception as e:
        print(f"   ❌ 实例化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 初始化 Opus 编码器（模拟 open_audio_channels）
    print("[2/4] 初始化音频通道 ...")
    from core.utils import opus_encoder_utils
    provider.opus_encoder = opus_encoder_utils.OpusEncoderUtils(
        sample_rate=conn.sample_rate, channels=1, frame_size_ms=60
    )
    print("   ✅ Opus 编码器已初始化")

    # 启动会话
    print("[3/4] 启动 TTS 会话 ...")
    try:
        await provider.start_session("test-session-001")
        print("   ✅ 会话启动成功")
    except Exception as e:
        print(f"   ❌ 会话启动失败: {e}")
        sys.exit(1)

    # 发送文本
    print("[4/4] 发送沪语测试文本 ...")
    test_text = "侬好，我是芯芯，老开心认得侬。"
    try:
        await provider.text_to_speak(test_text, None)
        print(f"   ✅ 文本已发送: {test_text}")
    except Exception as e:
        print(f"   ❌ 文本发送失败: {e}")
        await provider.close()
        sys.exit(1)

    # 等待音频收集
    print("   等待音频收集 (3s) ...")
    await asyncio.sleep(3)

    # 检查音频队列
    audio_count = 0
    while not provider.tts_audio_queue.empty():
        audio_count += 1
        try:
            provider.tts_audio_queue.get_nowait()
        except:
            break

    print()
    print("=" * 60)
    print("  验证结果")
    print("=" * 60)
    print(f"  Provider: {'✅' if provider else '❌'} 实例化")
    print(f"  WebSocket: {'✅' if provider.ws else '❌'} 连接")
    print(f"  文本发送: ✅")
    print(f"  音频队列: {audio_count} 条 {'✅' if audio_count > 0 else '⚠️ 可能无音频'}")

    # 清理
    await provider.close()
    print()
    print("  ⚠️ 完整端到端验证需连接 ESP32 设备进行实际对话测试")
    print("=" * 60)


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_provider())


if __name__ == "__main__":
    main()
