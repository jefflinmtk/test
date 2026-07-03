"""
Test 2: Concurrent Throughput (Batch / Multi-user)
- Sends N requests concurrently using asyncio
- Measures total throughput across all concurrent users
- Sweeps concurrency levels: 1, 2, 4, 8, 16
"""

import asyncio
import time
import argparse
import json
import httpx

DEFAULT_BASE_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL    = "Qwen/Qwen3.6-27B"

PROMPT = "Explain quantum computing in 3 paragraphs."

async def single_async_request(client: httpx.AsyncClient, url, model, prompt, max_tokens):
    payload = {
        "model":      model,
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream":     False,
    }
    t0 = time.perf_counter()
    resp = await client.post(url, json=payload, timeout=120)
    t1 = time.perf_counter()
    data = resp.json()

    usage = data.get("usage", {})
    return {
        "latency_s":    round(t1 - t0, 3),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens":  usage.get("total_tokens", 0),
    }


async def run_concurrent(url, model, concurrency, max_tokens):
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        tasks = [
            single_async_request(client, url, model, PROMPT, max_tokens)
            for _ in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    total_output_tokens = sum(r["output_tokens"] for r in results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    throughput_tps = total_output_tokens / wall_time

    return {
        "concurrency":       concurrency,
        "wall_time_s":       round(wall_time, 3),
        "avg_latency_s":     round(avg_latency, 3),
        "total_output_tokens": total_output_tokens,
        "throughput_tok_s":  round(throughput_tps, 2),
    }


async def main():
    parser = argparse.ArgumentParser(description="Concurrent throughput test")
    parser.add_argument("--base-url",     default=DEFAULT_BASE_URL)
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens",   type=int, default=256)
    parser.add_argument("--concurrency",  nargs="+", type=int, default=[1, 2, 4, 8, 16])
    args = parser.parse_args()

    all_results = []
    print(f"{'Concurrency':>12} | {'Wall(s)':>8} | {'AvgLat(s)':>10} | {'Tok/s':>10}")
    print("-" * 50)

    for c in args.concurrency:
        res = await run_concurrent(args.base_url, args.model, c, args.max_tokens)
        all_results.append(res)
        print(f"{res['concurrency']:>12} | {res['wall_time_s']:>8} | "
              f"{res['avg_latency_s']:>10} | {res['throughput_tok_s']:>10}")

    with open("results_concurrent_throughput.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to results_concurrent_throughput.json")


if __name__ == "__main__":
    asyncio.run(main())
