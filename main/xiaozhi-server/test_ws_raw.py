"""Qwen3-TTS WebSocket realtime API probe (Step 1A.2)

Tests BOTH endpoints:
  A) /api-ws/v1/realtime (OpenAI-style realtime protocol)
  B) /api-ws/v1/inference/ (CosyVoice-style inference protocol)
"""
import asyncio, json, base64, time, sys, os
import websockets

API_KEY = "sk-efc0be40d76d4553b3b48b9e12947756"
VOICE = "Jada"
TEST_TEXT = "侬好，我是芯芯，老开心认得侬。"

# ── Test A: Realtime endpoint ─────────────────────────────────────────

async def test_realtime():
    """Test wss://.../api-ws/v1/realtime (model as header)"""
    url = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    print(f"\n--- 方案A: Realtime 端点 (model header) ---")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-DashScope-Model": "qwen3-tts-flash-realtime",
    }
    try:
        ws = await websockets.connect(url, additional_headers=headers)
        print("[A1] WebSocket 连接成功")
    except Exception as e:
        print(f"[A1] FAIL: {e}")
        return False

    update_msg = {
        "type": "session.update",
        "session": {
            "voice": VOICE, "mode": "server_commit",
            "sample_rate": 24000, "speech_rate": 0.9,
            "volume": 50, "audio_format": "pcm",
            "language_type": "Chinese",
        }
    }
    await ws.send(json.dumps(update_msg))

    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=8))
        etype = msg.get("type", "")
        print(f"    <- {etype}")
        if etype == "error":
            print(f"    FAIL: {msg.get('error', msg)}")
            await ws.close()
            return False
        if etype in ("session.created", "session.updated"):
            break

    await ws.send(json.dumps({"type": "input_text_buffer.append", "text": TEST_TEXT}))
    print("[A2] append_text 已发送")

    chunks = []
    try:
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=12))
            etype = msg.get("type", "")
            if etype == "response.audio.delta":
                chunks.append(base64.b64decode(msg["delta"]))
                if len(chunks) == 1:
                    print(f"    <- audio.delta 开始流式返回")
            elif etype == "response.audio.done":
                print(f"    <- audio.done ({len(chunks)} 分片)")
            elif etype == "response.done":
                break
            elif etype == "error":
                print(f"    FAIL: {msg.get('error', msg)}")
                break
            else:
                print(f"    <- {etype}")
    except asyncio.TimeoutError:
        print("    超时")
    await ws.close()

    if chunks:
        all_pcm = b"".join(chunks)
        with open("test_ws_jada.pcm", "wb") as f:
            f.write(all_pcm)
        print(f"[A3] PCM已保存 ({len(all_pcm)} bytes)")
        return True
    return False

# ── Test B: Inference endpoint ────────────────────────────────────────

async def test_inference():
    """Test wss://.../api-ws/v1/inference/ (CosyVoice-style task protocol)"""
    url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"
    print(f"\n--- 方案B: Inference 端点 (CosyVoice协议) ---")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-DashScope-DataInspection": "enable",
    }
    try:
        ws = await websockets.connect(url, additional_headers=headers)
        print("[B1] WebSocket 连接成功")
    except Exception as e:
        print(f"[B1] FAIL: {e}")
        return False

    sid = "test-qwen3-tts-001"
    run_msg = {
        "header": {
            "action": "run-task",
            "task_id": sid,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "model": "qwen3-tts-flash-2025-11-27",
            "parameters": {
                "text_type": "PlainText",
                "voice": VOICE,
                "format": "pcm",
                "sample_rate": 24000,
                "volume": 50,
                "rate": 0.9,
            },
            "input": {}
        },
    }
    await ws.send(json.dumps(run_msg))
    print("[B2] run-task 已发送")

    # Wait for task-started
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        header = msg.get("header", {})
        event = header.get("event", "")
        print(f"    <- {event}")
        if event == "task-started":
            break
        elif event == "task-failed":
            print(f"    FAIL: {header.get('error_message', msg)}")
            await ws.close()
            return False

    # Send text
    continue_msg = {
        "header": {
            "action": "continue-task",
            "task_id": sid,
            "streaming": "duplex",
        },
        "payload": {"input": {"text": TEST_TEXT}},
    }
    await ws.send(json.dumps(continue_msg))

    # Finish
    finish_msg = {
        "header": {
            "action": "finish-task",
            "task_id": sid,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }
    await ws.send(json.dumps(finish_msg))
    print("[B3] text + finish 已发送")

    pcm_chunks = []
    try:
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=15)
            if isinstance(msg, (bytes, bytearray)):
                pcm_chunks.append(msg)
                if len(pcm_chunks) == 1:
                    print(f"    <- PCM 音频流开始")
            elif isinstance(msg, str):
                data = json.loads(msg)
                event = data.get("header", {}).get("event", "")
                if event == "result-generated":
                    print(f"    <- result-generated")
                elif event == "task-finished":
                    print(f"    <- task-finished")
                    break
                elif event == "task-failed":
                    print(f"    FAIL: {data.get('header',{}).get('error_message')}")
                    break
    except asyncio.TimeoutError:
        print("    超时")

    await ws.close()

    if pcm_chunks:
        all_pcm = b"".join(pcm_chunks)
        with open("test_inf_jada.pcm", "wb") as f:
            f.write(all_pcm)
        print(f"[B4] PCM已保存 ({len(all_pcm)} bytes, {len(pcm_chunks)} chunks)")
        return True
    return False

# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  Qwen3-TTS WebSocket 探路 (Step 1A.2)")
    print("=" * 60)

    ok_a = await test_realtime()
    ok_b = await test_inference()

    print()
    print("=" * 60)
    print(f"  方案A (realtime): {'PASS' if ok_a else 'FAIL'}")
    print(f"  方案B (inference): {'PASS' if ok_b else 'FAIL'}")
    if ok_a or ok_b:
        print("  RESULT: WebSocket 流式可用")
    else:
        print("  RESULT: WebSocket 流式不可用 -> 退回1B HTTP")
    print("=" * 60)

asyncio.run(main())
