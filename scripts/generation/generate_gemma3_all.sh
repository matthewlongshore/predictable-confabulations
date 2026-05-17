#!/bin/bash
# Generate Gemma 3 references for all 3 sizes (4B, 12B, 27B)
# Sequential to avoid rate limiting on OpenRouter free tier
# Temperature 0, nogeo prompt, 24 topics × 10 refs each

# Set OPENROUTER_API_KEY in your environment before running. Get a key at https://openrouter.ai/keys
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set}"

echo "=== Gemma 3 4B ==="
python3 scripts/generate_references_openrouter_gemma3_4b.py

echo ""
echo "=== Gemma 3 12B ==="
python3 scripts/generate_references_openrouter_gemma3_12b.py

echo ""
echo "=== Gemma 3 27B ==="
python3 scripts/generate_references_openrouter_gemma3_27b.py

echo ""
echo "=== ALL DONE ==="
