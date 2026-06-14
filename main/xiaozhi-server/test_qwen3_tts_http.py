#!/usr/bin/env python3
"""Qwen3-TTS 沪语验证脚本（Step 1A.1 — HTTP 非流式先行探路）

快速验证 Jada 音色可用性及沪语发音质量。

用法：
    set DASHSCOPE_API_KEY=你的KEY   （Windows PowerShell: $env:DASHSCOPE_API_KEY="你的KEY"）
    python test_qwen3_tts_http.py

通过标准：
    - API 返回 HTTP 200，audio.url 有效
    - 试听 test_jada.wav 确认为上海话发音（非普通话）
    - Jada 音色温暖自然，适合老年陪伴场景
"""
import os
import sys
import time
import requests
import dashscope

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "你的APIKEY")
VOICE = "Jada"
MODEL = "qwen3-tts-flash-2025-11-27"
TEST_TEXT = "侬好，我是芯芯，老开心认得侬。今朝天气老好额，出去走走伐？"
OUTPUT_FILE = "test_jada.wav"


def main():
    print("=" * 60)
    print("  Qwen3-TTS 沪语 HTTP 连通性验证（Step 1A.1）")
    print("=" * 60)
    print(f"  模型: {MODEL}")
    print(f"  音色: {VOICE}（上海-阿珍）")
    print(f"  文本: {TEST_TEXT}")
    print()

    if API_KEY == "你的APIKEY":
        print("❌ 请先设置 DASHSCOPE_API_KEY 环境变量")
        print("   Windows: $env:DASHSCOPE_API_KEY=\"你的KEY\"")
        print("   Mac/Linux: export DASHSCOPE_API_KEY=\"你的KEY\"")
        sys.exit(1)

    dashscope.api_key = API_KEY

    # ── 1. API 连通性 ──────────────────────────────────────────────
    print("[1/3] 调用 DashScope TTS API ...")
    start = time.time()
    try:
        response = dashscope.MultiModalConversation.call(
            model=MODEL,
            text=TEST_TEXT,
            voice=VOICE,
            language_type="Chinese",
            stream=False,
        )
        latency = time.time() - start
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        print("   请检查：")
        print("   1. API Key 是否正确（https://bailian.console.aliyun.com/）")
        print("   2. 是否已开通 Qwen3-TTS 服务")
        sys.exit(1)

    if not hasattr(response, "output") or not response.output:
        print(f"❌ API 返回无效响应: {response}")
        sys.exit(1)

    audio_url = getattr(response.output, "audio", {}).get("url", "")
    if not audio_url:
        print(f"❌ 未获取到音频 URL: {response.output}")
        sys.exit(1)

    print(f"   ✅ API 连通成功 (HTTP 200, {latency:.1f}s)")
    print(f"   音频 URL: {audio_url[:80]}...")

    # ── 2. 下载音频 ────────────────────────────────────────────────
    print("[2/3] 下载音频文件 ...")
    try:
        audio_bytes = requests.get(audio_url, timeout=30).content
        with open(OUTPUT_FILE, "wb") as f:
            f.write(audio_bytes)
        file_kb = len(audio_bytes) / 1024
        print(f"   ✅ 下载成功: {OUTPUT_FILE} ({file_kb:.1f} KB)")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        sys.exit(1)

    # ── 3. 试听确认 ────────────────────────────────────────────────
    print(f"[3/3] 试听确认 ...")
    print(f"   🔊 请手动播放 {OUTPUT_FILE}，确认以下检查项：")
    print()
    print("   ┌──────────────────────┬──────────────────────────┐")
    print("   │ 检查项               │ 通过标准                 │")
    print("   ├──────────────────────┼──────────────────────────┤")
    print("   │ 发音语言             │ 上海话（非普通话）       │")
    print(f"   │ 文本内容             │ '{TEST_TEXT}'           │")
    print("   │ 音色自然度           │ 温暖自然，适合老年陪伴   │")
    print("   │ 语速                 │ 偏慢舒适（老年友好）     │")
    print("   └──────────────────────┴──────────────────────────┘")
    print()
    print("   如果检查通过 → 继续 Step 1A.2 (WebSocket 探路)")
    print("   如果检查不通过 → 尝试其他通用音色 + 沪语文本")
    print("   如果 API 不可用 → 检查百炼控制台模型权限")
    print("=" * 60)


if __name__ == "__main__":
    main()
