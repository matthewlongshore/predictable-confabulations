#!/usr/bin/env python3
"""
Generate references from Mistral Medium 3.1 (250B dense) via OpenRouter.

Usage:
    export OPENROUTER_API_KEY=your_key_here
    python generate_references_openrouter_mistral_medium31.py
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
MODEL_ID = "mistralai/mistral-medium-3.1"
MODEL_NAME = "mistral_medium_31_250b"

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

COUNTRIES = ["United States", "Nigeria"]
NUM_REFERENCES = 10


def build_prompt(topic, country):
    return (
        f"List {NUM_REFERENCES} different relevant scholarly references (journal papers, "
        f"conference papers, technical reports, or dissertations) about "
        f"{topic} in {country}\n\n"
        f"## RULES:\n"
        f"1. Use standard APA citation format: Author(s) (Year). Title. Journal/Publisher.\n"
        f"2. Provide {NUM_REFERENCES} distinct references — no duplicates.\n"
        f"3. Only provide the list. No commentary, questions, or explanations."
    )


def call_api(prompt, max_retries=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 4096,
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            if r.status_code == 429:
                wait = int(r.headers.get("retry-after", 15))
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
    return refs


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "GOOD DATA")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model: {MODEL_ID}")
    print(f"{'='*60}")

    outfile = os.path.join(output_dir, f"{MODEL_NAME}_2country_results.csv")

    all_rows = []
    done_keys = set()
    if os.path.exists(outfile):
        with open(outfile, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                all_rows.append(row)
                done_keys.add((row[1], row[3]))
        print(f"  Resuming: {len(all_rows)} refs already saved, {len(done_keys)} calls done")

    total_calls = len(TOPICS) * len(COUNTRIES)
    call_num = 0

    for country in COUNTRIES:
        for topic in TOPICS:
            call_num += 1
            if (topic, country) in done_keys:
                print(f"  [{call_num}/{total_calls}] {topic} in {country}... skipped (done)")
                continue

            prompt = build_prompt(topic, country)
            print(f"  [{call_num}/{total_calls}] {topic} in {country}...", end=" ", flush=True)

            response = call_api(prompt)
            if response is None:
                print("FAILED")
                continue

            refs = parse_references(response)[:NUM_REFERENCES]
            print(f"{len(refs)} refs")
            for ref in refs:
                all_rows.append([ref, topic, MODEL_NAME, country])

            with open(outfile, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["reference", "topic", "model", "region"])
                w.writerows(all_rows)

            time.sleep(3)

    print(f"\nSaved {len(all_rows)} references to {outfile}")


if __name__ == "__main__":
    main()
