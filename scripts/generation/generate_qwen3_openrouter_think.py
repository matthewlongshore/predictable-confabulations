#!/usr/bin/env python3
"""
Generate references from Qwen3 dense family (8B, 14B) via OpenRouter.
Thinking ON (/think). No-geography prompt. Temperature 0. Resume-safe.
32B thinking already done via Groq.

Usage:
    export OPENROUTER_API_KEY=your_key_here
    python3 scripts/generate_qwen3_openrouter_think.py
"""

import os
import sys
import csv
import re
import time
import requests

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("Error: Set OPENROUTER_API_KEY environment variable")
    sys.exit(1)

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    ("qwen/qwen3-8b",  "qwen3_8b",  8),
    ("qwen/qwen3-14b", "qwen3_14b", 14),
]

TOPICS = [
    "Biometric voter registration",
    "Climate change",
    "Climate change adaptation in agriculture",
    "Climate-smart agriculture for smallholder farmers",
    "Democratic elections",
    "Digital financial services",
    "Economics",
    "Education",
    "Energy",
    "Environmental Science",
    "Girls education",
    "Health",
    "Infectious disease",
    "Insecticide-treated bed nets for malaria",
    "Malaria prevention",
    "Microfinance loan repayment",
    "Mini-grid electrification",
    "Mobile banking",
    "Political Science",
    "Renewable energy",
    "Rural electrification",
    "School dropout",
    "School dropout prevention programs in rural areas",
    "Voter turnout",
]

NUM_REFERENCES = 10


def build_prompt(topic):
    return (
        "/think\n"
        f"List {NUM_REFERENCES} different relevant scholarly references (journal papers, "
        f"conference papers, technical reports, or dissertations) about "
        f"{topic}\n\n"
        f"## RULES:\n"
        f"1. Use standard APA citation format: Author(s) (Year). Title. Journal/Publisher.\n"
        f"2. Provide {NUM_REFERENCES} distinct references — no duplicates.\n"
        f"3. Only provide the list. No commentary, questions, or explanations."
    )


def call_openrouter(model_id, prompt, max_retries=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/10x24",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 16384,
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=180)
            if r.status_code == 429:
                wait = int(r.headers.get("retry-after", 30))
                print(f"\n    Rate limited, waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 404:
                print(f"\n    Model {model_id} not available (404)")
                return None
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                print(f"\n    API error: {data['error']}", end=" ", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(10)
                continue
            text = data["choices"][0]["message"]["content"]
            # Strip any <think> blocks (some providers include them)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            # Log reasoning tokens if available
            usage = data.get("usage", {})
            reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            if reasoning:
                print(f"[{reasoning}r]", end=" ", flush=True)

            return text
        except requests.exceptions.RequestException as e:
            print(f"\n    Error (attempt {attempt+1}): {e}", end=" ", flush=True)
            if attempt < max_retries - 1:
                time.sleep(5)
    return None


def parse_references(text):
    if not text:
        return []
    refs = []
    current = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            if current:
                refs.append(" ".join(current))
                current = []
            continue
        is_numbered = False
        for i in range(1, NUM_REFERENCES + 5):
            for prefix in [f"{i}.", f"{i})", f"{i} .", f"{i} )"]:
                if line.startswith(prefix):
                    is_numbered = True
                    line = line[len(prefix):].strip()
                    break
            if is_numbered:
                break
        if line.startswith("- ") or line.startswith("• "):
            is_numbered = True
            line = line[2:].strip()
        if is_numbered and current:
            refs.append(" ".join(current))
            current = [line]
        elif line:
            current.append(line)
    if current:
        refs.append(" ".join(current))

    # Fallback: if only 1 ref but it's very long, try splitting on APA pattern
    if len(refs) <= 2 and refs and len(refs[0]) > 500:
        blob = refs[0]
        split_refs = re.split(r'(?<=[0-9.]) (?=[A-Z][a-z]+, [A-Z]\.)', blob)
        if len(split_refs) > 2:
            refs = [r.strip() for r in split_refs]

    return refs


def run_model(model_id, model_name):
    full_name = f"{model_name}_think"
    output_dir = os.path.join(os.path.dirname(__file__), "..", "GOOD DATA")
    outfile = os.path.join(output_dir, f"{full_name}_nogeo_results.csv")

    # Resume
    all_rows = []
    done_topics = set()
    if os.path.exists(outfile):
        with open(outfile, newline="") as f:
            for row in csv.DictReader(f):
                all_rows.append(row)
                done_topics.add(row["topic"])
        print(f"  Resuming: {len(all_rows)} refs, {len(done_topics)}/{len(TOPICS)} topics done")

    print(f"\n{'='*60}")
    print(f"  Model: {model_id} (THINKING)")
    print(f"  Output: {outfile}")
    print(f"{'='*60}")

    for i, topic in enumerate(TOPICS):
        if topic in done_topics:
            print(f"  [{i+1}/24] {topic} — skipped (done)")
            continue

        prompt = build_prompt(topic)
        print(f"  [{i+1}/24] {topic}...", end=" ", flush=True)

        response = call_openrouter(model_id, prompt)
        if response is None:
            print("FAILED")
            continue

        refs = parse_references(response)[:NUM_REFERENCES]
        print(f"{len(refs)} refs")

        for ref in refs:
            all_rows.append({
                "reference": ref,
                "topic": topic,
                "model": full_name,
            })
        done_topics.add(topic)

        with open(outfile, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["reference", "topic", "model"])
            w.writeheader()
            w.writerows(all_rows)

        time.sleep(3)

    print(f"  Done: {len(all_rows)} refs saved")
    return len(all_rows)


def main():
    print("Qwen3 Dense Family — THINKING ON (nogeo protocol, OpenRouter)")
    print(f"Models: 8B, 14B\n")

    total = 0
    for model_id, model_name, size_b in MODELS:
        total += run_model(model_id, model_name)

    print(f"\n{'='*60}")
    print(f"  ALL DONE: {total} total references")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
