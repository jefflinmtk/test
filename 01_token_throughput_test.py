"""
Test 1: Token Throughput & Latency
- Measures tokens/sec (generation throughput)
- Measures Time To First Token (TTFT)
- Measures end-to-end latency per request
"""

import time
import argparse
import json
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL    = "Qwen/Qwen3.6-27B"

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to sort a list using merge sort.",
    "What are the main differences between TCP and UDP?",
    "Summarize the history of artificial intelligence.",
    "Describe how transformers work in deep learning.",
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def single_request(client, model, prompt, max_tokens=256):
    t0 = time.perf_counter()
    first_token_time = None
    output_tokens = 0
    collected = []

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            collected.append(delta)
            output_tokens += 1  # approximate; 1 chunk ≈ 1 token for vLLM

    t1 = time.perf_counter()
    total_time = t1 - t0
    ttft = (first_token_time - t0) if first_token_time else None
    tps  = output_tokens / total_time if total_time > 0 else 0

    return {
        "prompt_preview": prompt[:60],
        "output_tokens":  output_tokens,
        "total_time_s":   round(total_time, 3),
        "ttft_s":         round(ttft, 3) if ttft else None,
        "tokens_per_sec": round(tps, 2),
        "response":       "".join(collected),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Token throughput & latency test")
    parser.add_argument("--base-url",   default=DEFAULT_BASE_URL)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeat",     type=int, default=1,
                        help="Repeat each prompt N times")
    args = parser.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=args.base_url)

    results = []
    for i, prompt in enumerate(PROMPTS):
        for r in range(args.repeat):
            print(f"\n[{i+1}/{len(PROMPTS)} repeat {r+1}] Sending: {prompt[:60]}...")
            res = single_request(client, args.model, prompt, args.max_tokens)
            results.append(res)
            print(f"  TTFT:       {res['ttft_s']} s")
            print(f"  Total time: {res['total_time_s']} s")
            print(f"  Output tok: {res['output_tokens']}")
            print(f"  Tok/sec:    {res['tokens_per_sec']}")

    # ── Summary ──
    avg_tps  = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_ttft = sum(r["ttft_s"] for r in results if r["ttft_s"]) / len(results)
    avg_lat  = sum(r["total_time_s"] for r in results) / len(results)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Requests run      : {len(results)}")
    print(f"  Avg tokens/sec    : {avg_tps:.2f}")
    print(f"  Avg TTFT          : {avg_ttft:.3f} s")
    print(f"  Avg total latency : {avg_lat:.3f} s")

    with open("results_token_throughput.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nDetailed results saved to results_token_throughput.json")


if __name__ == "__main__":
    main()
