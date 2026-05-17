#!/usr/bin/env python3
"""
Generate references from DeepSeek V4 Pro (MoE, 3.8B active / 26B total) via OpenRouter.
No-geography prompt. Temperature 0. Resume-safe.
"""

import os
import sys
import csv
import time
import requests

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("Error: Set OPENROUTER_API_KEY environment variable")
    sys.exit(1)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "deepseek/deepseek-v4-pro"
MODEL_NAME = "deepseek_v4_pro_1600b_a49b"

TOPICS = [
    "Biometric voter registration", "Climate change",
    "Climate change adaptation in agriculture",
    "Climate-smart agriculture for smallholder farmers",
    "Democratic elections", "Digital financial services",
    "Economics", "Education", "Energy", "Environmental Science",
    "Girls education", "Health", "Infectious disease",
    "Insecticide-treated bed nets for malaria", "Malaria prevention",
    "Microfinance loan repayment", "Mini-grid electrification",
    "Mobile banking", "Political Science", "Renewable energy",
    "Rural electrification", "School dropout",
    "School dropout prevention programs in rural areas", "Voter turnout",
]
NUM_REFERENCES = 10


def build_prompt(topic):
    return (
        f"List {NUM_REFERENCES} different relevant scholarly references (journal papers, "
        f"conference papers, technical reports, or dissertations) about "
        f"{topic}\n\n"
        f"## RULES:\n"
        f"1. Use standard APA citation format: Author(s) (Year). Title. Journal/Publisher.\n"
        f"2. Provide {NUM_REFERENCES} distinct references — no duplicates.\n"
        f"3. Only provide the list. No commentary, questions, or explanations."
    )


def call_api(prompt, max_retries=5):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_ID,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0}
    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=180)
            if r.status_code == 429:
                wait = int(r.headers.get("retry-after", 30))
                print(f"\n    Rate limited, waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"\n    Error (attempt {attempt+1}): {e}", end=" ", flush=True)
            if attempt < max_retries - 1:
                time.sleep(10)
    return None


def parse_references(text):
    if not text:
        return []
    refs, current = [], []
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
    return refs


def main():
    outdir = os.path.join(os.path.dirname(__file__), "..", "GOOD DATA")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{MODEL_NAME}_nogeo_results.csv")
    print(f"Model: {MODEL_ID}\nOutput: {outfile}\nTemperature: 0\n{'='*60}")

    all_rows, done = [], set()
    if os.path.exists(outfile):
        with open(outfile, newline="") as f:
            for row in csv.DictReader(f):
                all_rows.append(row)
                done.add(row["topic"])
        print(f"  Resuming: {len(all_rows)} refs, {len(done)}/{len(TOPICS)} topics")

    for i, topic in enumerate(TOPICS):
        if topic in done:
            print(f"  [{i+1}/{len(TOPICS)}] {topic}... skipped")
            continue
        print(f"  [{i+1}/{len(TOPICS)}] {topic}...", end=" ", flush=True)
        response = call_api(build_prompt(topic))
        if response is None:
            print("FAILED")
            continue
        refs = parse_references(response)[:NUM_REFERENCES]
        print(f"{len(refs)} refs")
        for ref in refs:
            all_rows.append({"reference": ref, "topic": topic, "model": MODEL_NAME})
        done.add(topic)
        with open(outfile, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["reference", "topic", "model"])
            w.writeheader()
            w.writerows(all_rows)
        time.sleep(2)
    print(f"\n{'='*60}\nDone: {len(all_rows)} references → {outfile}")


if __name__ == "__main__":
    main()
