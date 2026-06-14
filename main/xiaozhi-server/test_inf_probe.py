"""Qwen3-TTS Inference endpoint probe (CosyVoice-style protocol)"""
import asyncio, json, base64, time, sys, os
import websockets

API_KEY = "sk-efc0be40d76d4553b3b48b9e12947756"
VOICE = "Jada"
TEST_TEXT = "侬好，我是芯芯，老开心认得侬。"

# Test all model name variants
MODELS = [
    "qwen3-tts-flash-2025-11-27",
    "qwen3-tts-flash",
    "qwen-tts-flash",
    "qwen-tts",
    "cosyvoice-v3-flash",
]

async def test_model(model_name):
    """Test a single model on the inference endpoint"""
    url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-DashScope-DataInspection": "enable",
    }

    try:
        ws = await websockets.connect(url, additional_headers=headers, close_timeout=5)
    except Exception as e:
        print(f"  连接失败: {e}")
        return None

    sid = f"test-{model_name}-001"
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
            "model": model_name,
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

    # Wait for response
    try:
        msg = await asyncio.wait_for(ws.recv(), timeout=10)
        data = json.loads(msg)
        event = data.get("header", {}).get("event", "")
        if event == "task-started":
            print(f"   task-started! 继续...")

            # Send text and finish
            continue_msg = {
                "header": {"action": "continue-task", "task_id": sid, "streaming": "duplex"},
                "payload": {"input": {"text": TEST_TEXT}},
            }
            await ws.send(json.dumps(continue_msg))
            finish_msg = {
                "header": {"action": "finish-task", "task_id": sid, "streaming": "duplex"},
                "payload": {"input": {}},
            }
            await ws.send(json.dumps(finish_msg))

            # Collect audio
            pcm_chunks = []
            try:
                while True:
                    msg2 = await asyncio.wait_for(ws.recv(), timeout=15)
                    if isinstance(msg2, (bytes, bytearray)):
                        pcm_chunks.append(msg2)
                    elif isinstance(msg2, str):
                        data2 = json.loads(msg2)
                        evt = data2.get("header", {}).get("event", "")
                        if evt == "task-finished":
                            break
                        elif evt == "task-failed":
                            err = data2.get("header", {}).get("error_message", "?")
                            print(f"   合成失败: {err}")
                            break
            except asyncio.TimeoutError:
                pass

            await ws.close()

            if pcm_chunks:
                all_pcm = b"".join(pcm_chunks)
                fname = f"test_inf_{model_name}.pcm"
                with open(fname, "wb") as f:
                    f.write(all_pcm)
                print(f"   OK! {len(all_pcm)} bytes ({len(pcm_chunks)} chunks) -> {fname}")
                return True

        elif event == "task-failed":
            err = data.get("header", {}).get("error_message", "?")
            code = data.get("header", {}).get("error_code", "?")
            print(f"   FAIL: [{code}] {err}")
        else:
            print(f"   意外事件: {event}")
    except asyncio.TimeoutError:
        print("   超时")
    except Exception as e:
        print(f"   错误: {e}")

    await ws.close()
    return False

async def main():
    print("=" * 60)
    print("  Qwen3-TTS Inference 端点探路 (Step 1A.2 方案B)")
    print("=" * 60)

    for model in MODELS:
        print(f"\n  模型: {model} ...")
        result = await test_model(model)
        if result:
            print(f"\n  RESULT: {model} 可用!  WebSocket流式 PASS")
            return
        await asyncio.sleep(0.5)

    print(f"\n  RESULT: 所有模型均不可用 -> 退回 1B HTTP 方案")

asyncio.run(main())
