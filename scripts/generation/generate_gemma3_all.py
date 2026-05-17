#!/usr/bin/env python3
"""
Generate references from Gemma 3 (4B, 12B, 27B) via Google AI API.
No-geography prompt (topic only). Temperature 0. Resume-safe. Sequential.

Usage:
    export GOOGLE_GENERATIVE_AI_API_KEY=your_key
    python generate_gemma3_all.py
"""

import os
import sys
import csv
import re
import time
import requests

API_KEY = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
if not API_KEY:
    print("Error: Set GOOGLE_GENERATIVE_AI_API_KEY environment variable")
    sys.exit(1)

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

MODELS = [
    ("gemma-3-4b-it", "gemma_3_4b"),
    ("gemma-3-12b-it", "gemma_3_12b"),
    ("gemma-3-27b-it", "gemma_3_27b"),
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
        f"List {NUM_REFERENCES} different relevant scholarly references (journal papers, "
        f"conference papers, technical reports, or dissertations) about "
        f"{topic}\n\n"
        f"## RULES:\n"
        f"1. Use standard APA citation format: Author(s) (Year). Title. Journal/Publisher.\n"
        f"2. Provide {NUM_REFERENCES} distinct references — no duplicates.\n"
        f"3. Only provide the list. No commentary, questions, or explanations."
    )


def call_api(model_id, prompt, max_retries=5):
    url = f"{API_BASE}/{model_id}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 4096,
        },
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=120)
            if r.status_code == 429:
                wait = int(r.headers.get("retry-after", 30))
                wait = max(wait, 30)
                print(f"\n    Rate limited, waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text.strip()
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
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


def run_model(model_id, model_name):
    output_dir = os.path.join(os.path.dirname(__file__), "..", "GOOD DATA")
    outfile = os.path.join(output_dir, f"{model_name}_nogeo_results.csv")

    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"Output: {outfile}")
    print(f"Temperature: 0")
    print(f"{'='*60}")

    # Resume
    all_rows = []
    done_topics = set()
    if os.path.exists(outfile):
        with open(outfile, newline="") as f:
            for row in csv.DictReader(f):
                all_rows.append(row)
                done_topics.add(row["topic"])
        print(f"  Resuming: {len(all_rows)} refs, {len(done_topics)}/{len(TOPICS)} topics done")

    for i, topic in enumerate(TOPICS):
        if topic in done_topics:
            print(f"  [{i+1}/{len(TOPICS)}] {topic}... skipped (done)")
            continue

        prompt = build_prompt(topic)
        print(f"  [{i+1}/{len(TOPICS)}] {topic}...", end=" ", flush=True)

        response = call_api(model_id, prompt)
        if response is None:
            print("FAILED")
            continue

        refs = parse_references(response)[:NUM_REFERENCES]
        print(f"{len(refs)} refs")

        for ref in refs:
            all_rows.append({
                "reference": ref,
                "topic": topic,
                "model": model_name,
            })
        done_topics.add(topic)

        with open(outfile, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["reference", "topic", "model"])
            w.writeheader()
            w.writerows(all_rows)

        time.sleep(3)  # gentle on rate limits

    print(f"  Done: {len(all_rows)} refs saved")
    return len(all_rows)


def main():
    total = 0
    for model_id, model_name in MODELS:
        total += run_model(model_id, model_name)

    print(f"\n{'='*60}")
    print(f"ALL DONE: {total} total references across {len(MODELS)} models")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
