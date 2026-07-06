"""
Test 3: Context Length Stress Test
- Tests how throughput & latency scale with increasing input length
- Input sizes: 512, 1K, 2K, 4K, 8K tokens (prompt padded with repeated text)
"""

import time
import argparse
import json
from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL    = "Qwen/Qwen3.6-27B"

BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence is transforming every industry. "
    "Large language models can understand and generate human-like text. "
)

def build_prompt(target_words: int) -> str:
    repeated = (BASE_TEXT * ((target_words // len(BASE_TEXT.split())) + 2))
    words = repeated.split()[:target_words]
    body = " ".join(words)
    return body + "\n\nSummarize the above passage in one sentence."


def test_context(client, model, word_count, max_tokens):
    prompt = build_prompt(word_count)
    approx_tokens = word_count * 4 // 3  # rough word→token estimate

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=False,
    )
    elapsed = time.perf_counter() - t0

    usage = response.usage
    output_tok = usage.completion_tokens if usage else 0
    tps = output_tok / elapsed if elapsed > 0 else 0

    return {
        "approx_input_tokens": approx_tokens,
        "actual_prompt_tokens": usage.prompt_tokens if usage else None,
        "output_tokens":        output_tok,
        "total_time_s":         round(elapsed, 3),
        "tokens_per_sec":       round(tps, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Context length stress test")
    parser.add_argument("--base-url",   default=DEFAULT_BASE_URL)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=128)
    # word counts: approx 512 / 1K / 2K / 4K / 8K / 16k / 32K / 64K/ 128K /256K tokens
    parser.add_argument("--word-counts", nargs="+", type=int,
                        default=[384, 768, 1536, 3072, 6144, 6144*2, 6144*4, 6144*8, 6144*16, 6144*32])
    args = parser.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    results = []

    print(f"{'~Input Tok':>12} | {'Actual Input':>12} | {'Output Tok':>10} | "
          f"{'Time(s)':>8} | {'Tok/s':>8}")
    print("-" * 60)

    for wc in args.word_counts:
        print(f"  Testing ~{wc * 4 // 3} input tokens...", end="", flush=True)
        res = test_context(client, args.model, wc, args.max_tokens)
        results.append(res)
        print(f"\r{res['approx_input_tokens']:>12} | "
              f"{str(res['actual_prompt_tokens']):>12} | "
              f"{res['output_tokens']:>10} | "
              f"{res['total_time_s']:>8} | "
              f"{res['tokens_per_sec']:>8}")

    with open("results_context_length.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results_context_length.json")


if __name__ == "__main__":
    main()
