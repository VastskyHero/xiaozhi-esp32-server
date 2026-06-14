#!/usr/bin/env python3
"""Qwen3-TTS WebSocket 实时流式验证脚本（Step 1A.2）

验证 QwenTtsRealtime SDK 与项目的兼容性：
- WebSocket 连接 -> session.created
- 增量文本追加 -> response.audio.delta
- 延迟测量（首个 delta < 500ms）

用法：
    set DASHSCOPE_API_KEY=你的KEY
    python test_qwen3_tts_ws.py

通过标准：
    - WebSocket 连接成功，收到 session.created
    - 收到 response.audio.delta 事件，audio_chunks 非空
    - 试听 PCM 确认为沪语发音
    - 首个 response.audio.delta 在最终 append_text 后 < 500ms 到达
"""
import os
import sys
import time
import base64

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "你的APIKEY")
VOICE = "Jada"
MODEL = "qwen3-tts-flash-realtime"
TEST_PHRASES = [
    "侬好，",
    "我是芯芯，",
    "老开心认得侬。",
    "今朝天气老好额，",
    "阿拉出去走走伐？",
]
OUTPUT_FILE = "test_ws_jada.pcm"


class TestCallback:
    """收集音频 delta 并记录时间戳"""

    def __init__(self):
        self.audio_chunks = []
        self.events = []  # (timestamp, event_type)
        self.first_delta_time = None
        self.last_append_time = None

    def on_open(self):
        print("✅ WebSocket 连接已建立")

    def on_event(self, event: dict):
        now = time.time()
        etype = event.get("type", "")
        self.events.append((now, etype))

        if etype == "session.created":
            sid = event.get("session", {}).get("id", "unknown")
            print(f"✅ 会话创建: {sid}")
        elif etype == "session.updated":
            print("✅ 会话配置已应用")
        elif etype == "response.audio.delta":
            if self.first_delta_time is None:
                self.first_delta_time = now
                if self.last_append_time:
                    delta_ms = (now - self.last_append_time) * 1000
                    status = "✅" if delta_ms < 500 else "⚠️"
                    print(f"{status} 首个音频 delta 到达 (延迟: {delta_ms:.0f}ms)")
            try:
                self.audio_chunks.append(base64.b64decode(event["delta"]))
            except Exception as e:
                print(f"❌ base64 解码失败: {e}")
        elif etype == "response.audio.done":
            print(f"✅ 合成完成，共 {len(self.audio_chunks)} 个分片")
        elif etype == "error":
            print(f"❌ 服务端错误: {event}")

    def on_close(self, code, msg):
        print(f"🔒 连接关闭: code={code}")

    def on_error(self, error):
        print(f"❌ WebSocket 错误: {error}")


def main():
    print("=" * 60)
    print("  Qwen3-TTS WebSocket 实时流式验证（Step 1A.2）")
    print("=" * 60)
    print(f"  模型: {MODEL}")
    print(f"  音色: {VOICE}（上海-阿珍）")
    print(f"  模式: server_commit（服务端自动判断合成时机）")
    print(f"  文本: {''.join(TEST_PHRASES)}")
    print()

    if API_KEY == "你的APIKEY":
        print("❌ 请先设置 DASHSCOPE_API_KEY 环境变量")
        sys.exit(1)

    # ── 导入 QwenTtsRealtime ────────────────────────────────────────
    try:
        from dashscope.audio.qwen_tts import QwenTtsRealtime
        print("✅ dashscope.audio.qwen_tts.QwenTtsRealtime 导入成功")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("   请检查 dashscope 版本:")
        print("     pip show dashscope")
        print("   可能需要: pip install dashscope --upgrade")
        sys.exit(1)

    # ── WebSocket 连接 ─────────────────────────────────────────────
    cb = TestCallback()
    tts = QwenTtsRealtime(
        model=MODEL,
        api_key=API_KEY,
        callback=cb,
    )

    print("[1/4] 建立 WebSocket 连接 ...")
    try:
        tts.connect()
        print("   connect() 返回，后台线程已启动")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        sys.exit(1)

    # ── 会话配置 ───────────────────────────────────────────────────
    print("[2/4] 发送会话配置 (session.update) ...")
    try:
        tts.update_session(
            voice=VOICE,
            mode="server_commit",
            sample_rate=24000,
            speech_rate=0.9,
            volume=50,
            audio_format="pcm",
        )
        print("   update_session() 已发送")
    except Exception as e:
        print(f"❌ 配置失败: {e}")
        sys.exit(1)

    # 等待会话初始化
    time.sleep(1)

    # ── 模拟 LLM 流式输出 ──────────────────────────────────────────
    print("[3/4] 模拟 LLM 流式输出 (增量 append_text) ...")
    for i, phrase in enumerate(TEST_PHRASES):
        try:
            tts.append_text(phrase)
            cb.last_append_time = time.time()
            print(f"   [{i+1}/{len(TEST_PHRASES)}] append_text: {phrase}")
            time.sleep(0.3)  # 模拟 LLM token 间间隔
        except Exception as e:
            print(f"❌ append_text 失败: {e}")
            sys.exit(1)

    # ── 结束会话 ───────────────────────────────────────────────────
    print("[4/4] 结束会话 (session.finish) ...")
    try:
        tts.finish()
        print("   finish() 已发送")
    except Exception as e:
        print(f"❌ finish 失败: {e}")

    # 等待后台线程收集完音频
    print("   等待音频收集 (3s) ...")
    time.sleep(3)

    # ── 结果分析 ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  结果分析")
    print("=" * 60)

    total_pcm = sum(len(c) for c in cb.audio_chunks)
    print(f"  音频分片数: {len(cb.audio_chunks)}")
    print(f"  总 PCM 字节: {total_pcm} ({total_pcm / 24000:.1f}s @ 24kHz)")

    # 保存 PCM
    if cb.audio_chunks:
        with open(OUTPUT_FILE, "wb") as f:
            f.write(b"".join(cb.audio_chunks))
        print(f"  已保存: {OUTPUT_FILE}")

    # 延迟分析
    if cb.first_delta_time and cb.last_append_time:
        delta_ms = (cb.first_delta_time - cb.last_append_time) * 1000
        print(f"  首个 delta 延迟: {delta_ms:.0f}ms {'✅' if delta_ms < 500 else '⚠️ 偏高'}")

    print()
    print("  请试听 PCM 文件确认：")
    print(f"    ffplay -f s16le -ar 24000 -ac 1 {OUTPUT_FILE}")
    print("    或: python -c \"import pydub; pydub.AudioSegment(")
    print(f"        data=open('{OUTPUT_FILE}','rb').read(),")
    print("        sample_width=2, frame_rate=24000, channels=1")
    print("        ).export('test_ws_jada.wav', format='wav')\"")
    print()
    print("  ┌──────────────────────┬─────────────────────────────┐")
    print("  │ 检查项               │ 通过标准                    │")
    print("  ├──────────────────────┼─────────────────────────────┤")
    print("  │ WebSocket 连接       │ session.created 事件        │")
    print("  │ 音频 delta           │ audio_chunks 非空           │")
    print("  │ 沪语发音             │ 试听确认为上海话            │")
    print("  │ 延迟                 │ 首个 delta < 500ms          │")
    print("  │ SDK 兼容性           │ 无异常/崩溃                 │")
    print("  └──────────────────────┴─────────────────────────────┘")

    if cb.audio_chunks and cb.first_delta_time:
        if cb.first_delta_time - (cb.last_append_time or 0) < 0.5:
            print()
            print("  ✅ 全部通过 → 继续 Step 1A.3 (Provider 实现)")
        else:
            print()
            print("  ⚠️ 延迟超标 → 检查网络，或考虑退回 1B HTTP 方案")
    else:
        print()
        print("  ❌ 验证失败 → 检查 SDK 版本兼容性，考虑退回 1B HTTP 方案")

    print("=" * 60)


if __name__ == "__main__":
    main()
