#!/usr/bin/env python3
"""Fill blank K2 verdicts via OpenRouter (moonshotai/kimi-k2, same 0711 version)."""
import csv, os, sys, time, requests

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("Set OPENROUTER_API_KEY"); sys.exit(1)

CSV_PATH = "/Users/lilo/DATA/10x24/GOOD DATA/relevance_judge_k2_results.csv"
MODEL = "moonshotai/kimi-k2"
URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT = '''You are judging whether a scholarly paper is a relevant citation for a research topic.

TOPIC: {topic}
PAPER TITLE: {title}

Is this paper relevant to the topic?
- YES: directly and clearly about the topic
- PARTIAL: tangentially related or covers a broader/narrower aspect
- NO: not about the topic at all

Respond with exactly one word: YES, PARTIAL, or NO.'''


def judge(title, topic, max_retries=5):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "temperature": 0,
               "messages": [{"role": "user",
                             "content": PROMPT.format(topic=topic, title=title)}]}
    for attempt in range(max_retries):
        try:
            r = requests.post(URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 429:
                wait = int(r.headers.get("retry-after", 30))
                print(f" [429 wait {wait}s]", end="", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip().upper()
            for v in ["YES", "PARTIAL", "NO"]:
                if v in text:
                    return v
            return "UNKNOWN"
        except Exception as e:
            print(f" [err {attempt+1}: {str(e)[:60]}]", end="", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def main():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    pending = [i for i, r in enumerate(rows) if not r["verdict"].strip()]
    print(f"{len(rows)} total rows, {len(pending)} pending verdicts")

    done = 0
    for idx, i in enumerate(pending):
        r = rows[i]
        title = r["extracted_title"] or r["reference"][:200]
        v = judge(title, r["topic"])
        if v is None:
            print(f"\n  [{idx+1}/{len(pending)}] FAIL {r['model']} · {r['topic'][:30]}")
            continue
        rows[i]["verdict"] = v
        done += 1
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(pending)}] {v} — {r['model']} · {r['topic'][:35]}", flush=True)
        if done % 25 == 0:
            with open(CSV_PATH, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
        time.sleep(1.5)

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nFilled {done}/{len(pending)} verdicts.")


if __name__ == "__main__":
    main()
