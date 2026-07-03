"""
即時互動問答 (Interactive Chat)
- 在終端機直接和 Qwen3.6-27B 對話
- streaming 逐字輸出,體驗真實回應速度
- 保留多輪對話歷史 (記得前面說過的話)
- 每輪結束顯示 TTFT 與 tokens/sec

指令 (在提示符後輸入):
    /reset   清空對話歷史,重新開始
    /exit    離開 (或按 Ctrl+C)
"""

import time
import argparse
from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL    = "Qwen/Qwen3.6-27B"

SYSTEM_PROMPT = "You are a helpful assistant."


def stream_reply(client, model, messages, max_tokens, temperature):
    """送出對話,streaming 印出回應,回傳 (完整回應字串, 統計資料)。"""
    t0 = time.perf_counter()
    first_token_time = None
    token_count = 0
    chunks = []

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    print("\n\033[92mQwen:\033[0m ", end="", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            print(delta, end="", flush=True)
            chunks.append(delta)
            token_count += 1

    elapsed = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else 0
    tps = token_count / elapsed if elapsed > 0 else 0

    stats = {"ttft": ttft, "tps": tps, "tokens": token_count, "total": elapsed}
    return "".join(chunks), stats


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with Qwen3.6-27B")
    parser.add_argument("--base-url",    default=DEFAULT_BASE_URL)
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens",  type=int,   default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-stats",    action="store_true", help="不顯示速度統計")
    args = parser.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=args.base_url)

    print("=" * 60)
    print(f"  即時對話 — {args.model}")
    print(f"  server : {args.base_url}")
    print("  指令   : /reset 清空歷史 | /exit 離開 (或 Ctrl+C)")
    print("=" * 60)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\n\033[96m你:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見!")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("再見!")
            break
        if user_input == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("(已清空對話歷史)")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            reply, stats = stream_reply(
                client, args.model, messages,
                args.max_tokens, args.temperature,
            )
        except Exception as e:
            print(f"\n\033[91m[錯誤]\033[0m {e}")
            print("  → 確認 vLLM server 有啟動,且 --model 名稱正確")
            messages.pop()  # 移除這輪失敗的提問
            continue

        messages.append({"role": "assistant", "content": reply})

        if not args.no_stats:
            print(f"\n\033[90m  [TTFT {stats['ttft']:.2f}s | "
                  f"{stats['tokens']} tok | "
                  f"{stats['tps']:.1f} tok/s | "
                  f"共 {stats['total']:.2f}s]\033[0m")


if __name__ == "__main__":
    main()
