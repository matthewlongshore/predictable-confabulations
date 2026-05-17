#!/usr/bin/env python3
"""
Generate references from Llama 4 Maverick via Groq.
Same protocol, single model.

Usage:
    export GROQ_API_KEY=your_key_here
    python generate_references_groq_maverick.py
"""

import os
import sys
import csv
import time
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Error: Set GROQ_API_KEY environment variable")
    sys.exit(1)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"
MODEL_NAME = "llama_4_maverick"

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

COUNTRIES = ["United States", "Nigeria", "Ghana"]
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


def call_groq(prompt, max_retries=3):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
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
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 429:
                wait = int(r.headers.get("retry-after", 10))
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"    Error (attempt {attempt+1}): {e}")
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
    return refs


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "GOOD DATA")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model: {MODEL_ID}")
    print(f"{'='*60}")

    all_rows = []
    total_calls = len(TOPICS) * len(COUNTRIES)
    call_num = 0

    for country in COUNTRIES:
        for topic in TOPICS:
            call_num += 1
            prompt = build_prompt(topic, country)
            print(f"  [{call_num}/{total_calls}] {topic} in {country}...", end=" ", flush=True)

            response = call_groq(prompt)
            if response is None:
                print("FAILED")
                continue

            refs = parse_references(response)[:NUM_REFERENCES]
            print(f"{len(refs)} refs")
            for ref in refs:
                all_rows.append([ref, topic, MODEL_NAME, country])

            # Save after every call
            outfile = os.path.join(output_dir, f"{MODEL_NAME}_3country_results.csv")
            with open(outfile, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["reference", "topic", "model", "region"])
                w.writerows(all_rows)

            time.sleep(2.5)

    print(f"\nSaved {len(all_rows)} references to {outfile}")


if __name__ == "__main__":
    main()
