"""
TourismGPT Fine-tuning Pipeline
Converts 3 datasets into instruction-output JSONL pairs.

Datasets:
  1. TripAdvisor Hotel Reviews (local CSV)          → hotel booking + destination comparison
  2. NLPC-UOM/Travel-Dataset-5000 (HuggingFace)    → itinerary + budget + customs/safety
  3. Bitext Customer Support (HuggingFace)          → cancellation + refund support

Output: tourism_finetune.jsonl  (~1,200 clean pairs)
"""

import json
import ast
import re
import random
import pandas as pd
from datasets import load_dataset
from pathlib import Path

random.seed(42)

OUTPUT_FILE = "tourism_finetune.jsonl"
REVIEWS_CSV  = "archive/reviews.csv"
OFFERINGS_CSV = "archive/offerings.csv"

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def save_jsonl(pairs: list[dict], path: str):
    with open(path, "a", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(pairs)} pairs → {path}")


def word_count(text: str) -> int:
    return len(str(text).split())


# ─────────────────────────────────────────────
# Dataset 1 — TripAdvisor (local CSV)
# Intent: hotel booking help + destination comparison
# Target: 200 + 200 = 400 pairs
# ─────────────────────────────────────────────

def parse_ratings(raw) -> dict:
    try:
        return ast.literal_eval(str(raw))
    except Exception:
        return {}


def format_ratings(r: dict) -> str:
    labels = {
        "overall": "Overall", "service": "Service", "cleanliness": "Cleanliness",
        "value": "Value", "location": "Location", "rooms": "Rooms",
        "sleep_quality": "Sleep Quality"
    }
    parts = [f"{labels.get(k, k)}: {int(v)}/5" for k, v in r.items() if v]
    return ", ".join(parts)


def process_tripadvisor(target_per_intent=200) -> list[dict]:
    print("\n[1/3] Processing TripAdvisor reviews...")

    offerings = pd.read_csv(OFFERINGS_CSV, usecols=["id", "name", "hotel_class", "address"])
    offerings["id"] = offerings["id"].astype(str)
    hotel_map = offerings.set_index("id").to_dict("index")

    reviews = pd.read_csv(REVIEWS_CSV, usecols=["ratings", "text", "offering_id", "date_stayed"])
    reviews["offering_id"] = reviews["offering_id"].astype(str)
    reviews = reviews.dropna(subset=["text"])

    # Filter: minimum 80 words
    reviews = reviews[reviews["text"].apply(lambda t: word_count(t) >= 80)]
    reviews = reviews.merge(
        offerings.rename(columns={"id": "offering_id"}),
        on="offering_id", how="left"
    ).dropna(subset=["name"])

    pairs = []

    # --- Intent 1: Hotel Booking Help ---
    booking_templates = [
        ("Recommend a hotel with good {aspect} in {city}.",
         "Based on guest reviews, {name} ({stars}★) in {city} is highly praised for {aspect}. "
         "Guests noted: \"{snippet}\" Ratings — {ratings}."),
        ("Is {name} a good hotel to stay at?",
         "{name} ({stars}★) has received positive feedback from guests. "
         "A reviewer said: \"{snippet}\" Ratings — {ratings}."),
        ("What do guests say about {name}?",
         "Guests who stayed at {name} ({stars}★) shared the following: \"{snippet}\" "
         "Overall ratings — {ratings}."),
    ]

    aspects = ["cleanliness", "location", "service", "value", "rooms"]

    sample = reviews.sample(min(target_per_intent * 5, len(reviews)))
    count = 0
    for _, row in sample.iterrows():
        if count >= target_per_intent:
            break
        r = parse_ratings(row["ratings"])
        if not r or r.get("overall", 0) < 3.5:
            continue
        snippet = str(row["text"])[:300].strip().replace("\n", " ")
        address = row.get("address", "")
        city = ""
        try:
            addr = ast.literal_eval(str(address))
            city = addr.get("locality", "")
        except Exception:
            pass
        if not city:
            continue

        aspect = max(aspects, key=lambda a: r.get(a, 0))
        stars = int(float(row.get("hotel_class") or 3) if pd.notna(row.get("hotel_class")) else 3)
        tmpl_i, tmpl_o = random.choice(booking_templates)

        pairs.append({
            "instruction": tmpl_i.format(
                aspect=aspect, city=city, name=row["name"], stars=stars
            ),
            "output": tmpl_o.format(
                name=row["name"], stars=stars, city=city,
                aspect=aspect, snippet=snippet, ratings=format_ratings(r)
            ),
            "intent": "hotel_booking_help"
        })
        count += 1

    # --- Intent 2: Destination Comparison ---
    city_groups = {}
    for _, row in reviews.iterrows():
        try:
            addr = ast.literal_eval(str(row.get("address", "")))
            city = addr.get("locality", "")
        except Exception:
            city = ""
        if city:
            city_groups.setdefault(city, []).append(row)

    cities = [c for c, rows in city_groups.items() if len(rows) >= 5]
    count = 0
    for _ in range(target_per_intent * 10):
        if count >= target_per_intent or len(cities) < 2:
            break
        c1, c2 = random.sample(cities, 2)
        r1 = random.choice(city_groups[c1])
        r2 = random.choice(city_groups[c2])
        rat1 = parse_ratings(r1["ratings"])
        rat2 = parse_ratings(r2["ratings"])
        if not rat1 or not rat2:
            continue
        overall1 = rat1.get("overall", 0)
        overall2 = rat2.get("overall", 0)
        better = c1 if overall1 >= overall2 else c2
        pairs.append({
            "instruction": f"Compare hotels in {c1} vs {c2} for a family trip.",
            "output": (
                f"Both {c1} and {c2} are popular destinations. "
                f"Hotels in {c1} score an average overall rating of {overall1}/5, "
                f"while {c2} hotels average {overall2}/5. "
                f"For families, {better} tends to offer better value and comfort based on guest feedback. "
                f"A guest in {c1} noted: \"{str(r1['text'])[:200].strip()}\" "
                f"A guest in {c2} noted: \"{str(r2['text'])[:200].strip()}\""
            ),
            "intent": "destination_comparison"
        })
        count += 1

    print(f"  → TripAdvisor: {len(pairs)} pairs generated")
    return pairs


# ─────────────────────────────────────────────
# Dataset 2 — NLPC-UOM Travel-Dataset-5000
# Intent: itinerary planning + budget estimation + local customs/safety
# Target: 200 + 200 + 200 = 600 pairs (split by keyword matching)
# ─────────────────────────────────────────────

ITINERARY_KEYWORDS = [
    "itinerary", "day trip", "schedule", "plan", "visit", "tour",
    "day 1", "days in", "how long", "route", "trip to"
]
BUDGET_KEYWORDS = [
    "budget", "cost", "price", "cheap", "expensive", "afford",
    "how much", "money", "spend", "fee", "ticket"
]
CUSTOMS_KEYWORDS = [
    "custom", "culture", "tradition", "etiquette", "dress code",
    "safe", "safety", "crime", "tip", "local", "religion", "law"
]


def classify_travel_question(text: str):
    t = text.lower()
    if any(k in t for k in ITINERARY_KEYWORDS):
        return "itinerary_planning"
    if any(k in t for k in BUDGET_KEYWORDS):
        return "budget_estimation"
    if any(k in t for k in CUSTOMS_KEYWORDS):
        return "local_customs_safety"
    return None


def process_travel_dataset(target_per_intent=200) -> list[dict]:
    print("\n[2/3] Processing NLPC-UOM Travel-Dataset-5000...")

    # CSV has non-UTF-8 chars — download raw file and read with latin-1
    from huggingface_hub import hf_hub_download
    csv_path = hf_hub_download(
        repo_id="NLPC-UOM/Travel-Dataset-5000",
        filename="5000TravelQuestionsDataset.csv",
        repo_type="dataset"
    )
    # No header row — first row is actual data; columns are: question, category_code, intent_code
    df = pd.read_csv(csv_path, encoding="latin-1", on_bad_lines="skip",
                     header=None, names=["question", "category", "intent_code"])

    # Restore the first row that pandas ate as header
    first_row = pd.DataFrame([[csv_path, None, None]], columns=df.columns)  # placeholder
    # Re-read properly: the "column name" that got eaten is the first question
    eaten_question = pd.read_csv(csv_path, encoding="latin-1", nrows=0).columns[0]
    first_row = pd.DataFrame([[eaten_question, None, None]], columns=df.columns)
    df = pd.concat([first_row, df], ignore_index=True)

    print(f"  Rows loaded: {len(df)} | Sample: {df['question'].iloc[0][:80]}")

    q_col = "question"
    a_col = None  # no answer column — will use fallback templates

    buckets = {"itinerary_planning": [], "budget_estimation": [], "local_customs_safety": []}

    for _, row in df.iterrows():
        question = str(row[q_col]).strip()
        intent = classify_travel_question(question)
        if not intent:
            continue
        answer = str(row[a_col]).strip() if a_col else ""
        if len(answer) < 20:
            continue
        buckets[intent].append({
            "instruction": question,
            "output": answer,
            "intent": intent
        })

    pairs = []
    for intent, items in buckets.items():
        selected = random.sample(items, min(target_per_intent, len(items)))
        pairs.extend(selected)
        print(f"  → {intent}: {len(selected)} pairs")

    # If answer column not found, generate templated outputs from question alone
    if not a_col:
        print("  ⚠ No answer column found — using question-only fallback templates")
        pairs = _fallback_travel_templates(df, q_col, target_per_intent)

    return pairs


def _fallback_travel_templates(df, q_col, target_per_intent) -> list[dict]:
    """Used if Travel-Dataset-5000 has no answer column."""
    templates = {
        "itinerary_planning": (
            "Here is a suggested itinerary based on your query: {q} "
            "Day 1: Arrive and explore the city center. Day 2: Visit major landmarks. "
            "Day 3: Day trip to nearby attractions. Adjust based on your pace and interests."
        ),
        "budget_estimation": (
            "For your query '{q}': Budget travellers can expect to spend $50–80/day covering "
            "accommodation, meals, and transport. Mid-range budgets of $100–150/day allow more comfort. "
            "Always set aside 10–15% for unexpected expenses."
        ),
        "local_customs_safety": (
            "Regarding '{q}': Research local customs before visiting. Dress modestly at religious sites, "
            "respect local traditions, and always check travel advisories from your government. "
            "Keep copies of important documents and stay aware of your surroundings."
        ),
    }
    pairs = []
    for _, row in df.iterrows():
        question = str(row[q_col]).strip()
        intent = classify_travel_question(question)
        if not intent:
            continue
        pairs.append({
            "instruction": question,
            "output": templates[intent].format(q=question),
            "intent": intent
        })
    buckets = {}
    for p in pairs:
        buckets.setdefault(p["intent"], []).append(p)
    result = []
    for intent, items in buckets.items():
        result.extend(random.sample(items, min(target_per_intent, len(items))))
    return result


# ─────────────────────────────────────────────
# Dataset 3 — Bitext Customer Support
# Intent: cancellation + refund support
# Target: 200 + 200 = 400 pairs
# ─────────────────────────────────────────────

BITEXT_TRAVEL_INTENTS = {
    "cancel_order":      "cancellation_refund_support",
    "cancellation_fee":  "cancellation_refund_support",
    "get_refund":        "cancellation_refund_support",
    "track_refund":      "cancellation_refund_support",
    "change_order":      "cancellation_refund_support",
    "place_order":       "hotel_booking_help",
    "delivery_options":  "hotel_booking_help",
    "payment_issue":     "cancellation_refund_support",
    "check_invoice":     "cancellation_refund_support",
    "review":            "hotel_booking_help",
    "complaint":         "cancellation_refund_support",
}

TRAVEL_REWRITE = {
    "order":    "booking",
    "product":  "reservation",
    "item":     "room",
    "package":  "travel package",
    "store":    "hotel",
    "shop":     "travel agency",
    "purchase": "booking",
    "bought":   "booked",
    "buy":      "book",
}


def travel_rewrite(text: str) -> str:
    for generic, travel in TRAVEL_REWRITE.items():
        text = re.sub(rf"\b{generic}\b", travel, text, flags=re.IGNORECASE)
    return text


def process_bitext(target_per_intent=200) -> list[dict]:
    print("\n[3/3] Processing Bitext Customer Support...")

    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train"
    )
    df = ds.to_pandas()

    print(f"  Available intents: {sorted(df['intent'].unique())}")

    # Filter to travel-relevant intents only
    relevant = df[df["intent"].isin(BITEXT_TRAVEL_INTENTS.keys())].copy()
    relevant["intent_mapped"] = relevant["intent"].map(BITEXT_TRAVEL_INTENTS)
    relevant["instruction"] = relevant["instruction"].apply(travel_rewrite)
    relevant["response"]    = relevant["response"].apply(travel_rewrite)

    pairs = []
    for mapped_intent in relevant["intent_mapped"].unique():
        subset = relevant[relevant["intent_mapped"] == mapped_intent]
        sample = subset.sample(min(target_per_intent, len(subset)), random_state=42)
        for _, row in sample.iterrows():
            pairs.append({
                "instruction": str(row["instruction"]).strip(),
                "output":      str(row["response"]).strip(),
                "intent":      mapped_intent
            })
        print(f"  → {mapped_intent}: {min(target_per_intent, len(subset))} pairs")

    return pairs


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # Clear previous output
    Path(OUTPUT_FILE).unlink(missing_ok=True)

    all_pairs = []

    pairs_1 = process_tripadvisor(target_per_intent=200)
    all_pairs.extend(pairs_1)

    pairs_2 = process_travel_dataset(target_per_intent=200)
    all_pairs.extend(pairs_2)

    pairs_3 = process_bitext(target_per_intent=200)
    all_pairs.extend(pairs_3)

    # Shuffle before saving
    random.shuffle(all_pairs)
    save_jsonl(all_pairs, OUTPUT_FILE)

    # Summary
    print("\n" + "═" * 50)
    print("PIPELINE COMPLETE")
    print("═" * 50)

    from collections import Counter
    intent_counts = Counter(p["intent"] for p in all_pairs)
    for intent, count in sorted(intent_counts.items()):
        status = "✅" if count >= 150 else "⚠️ "
        print(f"  {status} {intent:<35} {count} pairs")

    print(f"\n  Total pairs : {len(all_pairs)}")
    print(f"  Output file : {OUTPUT_FILE}")
    print("═" * 50)


if __name__ == "__main__":
    main()
