"""
AI-Generated Fine-tuning Pairs for TourismGPT
Generates ~1,000 high-quality, diverse instruction-output pairs
across all 6 intent categories and appends to tourism_finetune.jsonl.

Target: bring total from 1,380 → ~2,380 pairs
"""

import json
import random
from pathlib import Path
from itertools import product

random.seed(99)

OUTPUT_FILE = "tourism_finetune.jsonl"

# ─────────────────────────────────────────────
# Shared Data
# ─────────────────────────────────────────────

DESTINATIONS = [
    ("Japan", "Asia", "Tokyo, Kyoto, Osaka"),
    ("Italy", "Europe", "Rome, Florence, Venice"),
    ("Thailand", "Asia", "Bangkok, Chiang Mai, Phuket"),
    ("France", "Europe", "Paris, Nice, Lyon"),
    ("Australia", "Oceania", "Sydney, Melbourne, Cairns"),
    ("Mexico", "Americas", "Mexico City, Cancun, Oaxaca"),
    ("Morocco", "Africa", "Marrakech, Fes, Casablanca"),
    ("Peru", "Americas", "Lima, Cusco, Machu Picchu"),
    ("Greece", "Europe", "Athens, Santorini, Mykonos"),
    ("India", "Asia", "Delhi, Agra, Jaipur"),
    ("New Zealand", "Oceania", "Auckland, Queenstown, Rotorua"),
    ("Portugal", "Europe", "Lisbon, Porto, Algarve"),
    ("Vietnam", "Asia", "Hanoi, Ho Chi Minh City, Ha Long Bay"),
    ("South Africa", "Africa", "Cape Town, Johannesburg, Kruger National Park"),
    ("Canada", "Americas", "Toronto, Vancouver, Banff"),
    ("Turkey", "Europe/Asia", "Istanbul, Cappadocia, Antalya"),
    ("Egypt", "Africa", "Cairo, Luxor, Aswan"),
    ("Argentina", "Americas", "Buenos Aires, Mendoza, Patagonia"),
    ("Iceland", "Europe", "Reykjavik, Golden Circle, Blue Lagoon"),
    ("Indonesia", "Asia", "Bali, Jakarta, Lombok"),
    ("Spain", "Europe", "Madrid, Barcelona, Seville"),
    ("Sri Lanka", "Asia", "Colombo, Sigiriya, Galle"),
    ("Kenya", "Africa", "Nairobi, Masai Mara, Mombasa"),
    ("Colombia", "Americas", "Bogota, Cartagena, Medellin"),
    ("Croatia", "Europe", "Zagreb, Dubrovnik, Split"),
]

DURATIONS = [3, 5, 7, 10, 14]
BUDGETS   = ["$500", "$1,000", "$1,500", "$2,000", "$3,000", "$5,000", "$8,000"]
TRAVELER_TYPES = ["solo traveler", "couple", "family with kids", "group of friends", "honeymoon couple", "senior travelers", "backpacker"]
SEASONS   = ["January", "March", "June", "July", "October", "December"]

# ─────────────────────────────────────────────
# Intent 1: Itinerary Planning (~200 pairs)
# ─────────────────────────────────────────────

ITINERARY_TEMPLATES = [
    {
        "instruction": "Suggest a {days}-day itinerary for {country} for a {traveler}.",
        "output": (
            "Here is a {days}-day itinerary for {country} tailored for a {traveler}:\n\n"
            "Day 1–2: Arrive in {city1}. Check into your hotel, recover from jet lag, and explore the city center. "
            "Visit the main landmarks and enjoy local street food.\n\n"
            "Day 3–{mid}: Head to {city2}. Spend time exploring historical sites, markets, and local neighborhoods. "
            "Book a half-day guided tour for deeper cultural immersion.\n\n"
            "Day {mid2}–{end}: Travel to {city3} for the final stretch. "
            "Relax, enjoy nature or beach activities, and do any last-minute souvenir shopping.\n\n"
            "Day {days}: Depart. Allow at least 3 hours before your flight for airport transfer and check-in.\n\n"
            "Tips for {traveler}s: Book accommodations in advance, carry local currency, and always have a translation app handy."
        )
    },
    {
        "instruction": "Plan a {days}-day trip to {country} in {season}. I'm a {traveler}.",
        "output": (
            "{season} is a {season_note} time to visit {country}. Here is your {days}-day plan:\n\n"
            "Days 1–2 ({city1}): Arrive and acclimatize. Explore the old town and visit top-rated attractions. "
            "Try the local cuisine at a well-reviewed restaurant.\n\n"
            "Days 3–{mid} ({city2}): Take a day trip or local bus. Visit museums, temples, or natural parks. "
            "This is the cultural heart of your trip.\n\n"
            "Days {mid2}–{days} ({city3}): Wind down. Enjoy scenery, shopping, or adventure activities "
            "depending on your pace. Ideal for a {traveler}.\n\n"
            "Packing tip for {season}: {packing_tip}."
        )
    },
    {
        "instruction": "What is the best {days}-day route through {country}?",
        "output": (
            "The classic {days}-day route through {country} flows through {cities}.\n\n"
            "Start in {city1} — spend 2 days here taking in the major sights and adjusting to the local timezone. "
            "Then move to {city2} for {mid_days} days, which is renowned for its {city2_highlight}. "
            "End your trip in {city3}, perfect for winding down before your departure.\n\n"
            "Transportation: Trains or budget airlines connect these cities efficiently. "
            "Book tickets at least a week in advance to get the best fares.\n\n"
            "Best suited for: {traveler}s who want a balanced mix of culture, food, and sightseeing."
        )
    },
    {
        "instruction": "I have {days} days in {country}. What should I prioritize?",
        "output": (
            "With {days} days in {country}, focus on quality over quantity. Here is what to prioritize:\n\n"
            "Must-do: {city1} for its iconic landmarks and vibrant street life. "
            "Do not miss {city2} for cultural depth — it represents the heart of {country}'s heritage.\n\n"
            "Optional: If time allows, a quick visit to {city3} adds variety, especially for nature lovers.\n\n"
            "Avoid: Trying to cover too many destinations in {days} days leads to exhaustion. "
            "Pick two cities and explore them thoroughly rather than rushing through five.\n\n"
            "Local tip: {local_tip}."
        )
    },
]

SEASON_NOTES = {
    "January": "a cooler, off-peak", "March": "a pleasant spring", "June": "a warm, popular",
    "July": "a peak summer", "October": "a mild autumn", "December": "a festive winter"
}
PACKING_TIPS = {
    "January": "Pack layers — mornings can be cold even in tropical regions",
    "March": "Light clothing with a jacket for evenings",
    "June": "Sunscreen, light breathable fabrics, and a rain jacket",
    "July": "Pack for heat and crowds — carry a reusable water bottle",
    "October": "A light sweater and waterproof layer work well",
    "December": "Festive season — book everything early and pack warm layers"
}
CITY_HIGHLIGHTS = ["ancient temples", "world-class museums", "stunning coastline", "vibrant night markets", "UNESCO heritage sites", "dramatic mountain scenery"]
LOCAL_TIPS = [
    "Download an offline map before you arrive",
    "Learn 5 basic phrases in the local language — locals appreciate the effort",
    "Carry small denomination bills for street vendors and taxis",
    "Travel insurance is strongly recommended for medical emergencies",
    "Book popular attractions in advance — queues can be hours long",
]


def gen_itinerary(n=200):
    pairs = []
    combos = list(product(DESTINATIONS, DURATIONS, TRAVELER_TYPES, SEASONS))
    random.shuffle(combos)
    for (dest, dur, trav, season) in combos[:n]:
        country, region, cities_str = dest
        cities = [c.strip() for c in cities_str.split(",")]
        city1 = cities[0]
        city2 = cities[1] if len(cities) > 1 else cities[0]
        city3 = cities[2] if len(cities) > 2 else cities[0]
        mid = max(3, dur // 2)
        mid2 = mid + 1
        tmpl = random.choice(ITINERARY_TEMPLATES)
        instruction = tmpl["instruction"].format(
            days=dur, country=country, traveler=trav, season=season
        )
        output = tmpl["output"].format(
            days=dur, country=country, traveler=trav, season=season,
            city1=city1, city2=city2, city3=city3, cities=cities_str,
            mid=mid, mid2=mid2, end=dur,
            mid_days=max(2, dur - mid - 1),
            season_note=SEASON_NOTES.get(season, "good"),
            packing_tip=PACKING_TIPS.get(season, "Pack light and versatile clothing"),
            city2_highlight=random.choice(CITY_HIGHLIGHTS),
            local_tip=random.choice(LOCAL_TIPS),
        )
        pairs.append({"instruction": instruction, "output": output, "intent": "itinerary_planning"})
    return pairs


# ─────────────────────────────────────────────
# Intent 2: Destination Comparison (~150 pairs)
# ─────────────────────────────────────────────

COMPARISON_TEMPLATES = [
    {
        "instruction": "{dest1} vs {dest2} — which is better for a {traveler}?",
        "output": (
            "Both {dest1} and {dest2} are excellent choices, but they suit different travel styles.\n\n"
            "{dest1} ({region1}) is ideal if you prioritize {strength1}. "
            "It is particularly well-suited for {best_for1}.\n\n"
            "{dest2} ({region2}) shines when it comes to {strength2}. "
            "Travellers who love {best_for2} tend to prefer it.\n\n"
            "For a {traveler}: {recommendation}.\n\n"
            "Budget note: {dest1} typically costs {cost1} per day while {dest2} averages {cost2} per day all-in."
        )
    },
    {
        "instruction": "We are choosing between {dest1} and {dest2} for our vacation. Help us decide.",
        "output": (
            "Great shortlist! Here is a side-by-side comparison:\n\n"
            "| Factor | {dest1} | {dest2} |\n"
            "|---|---|---|\n"
            "| Region | {region1} | {region2} |\n"
            "| Best for | {best_for1} | {best_for2} |\n"
            "| Avg daily cost | {cost1} | {cost2} |\n"
            "| Ease of travel | {ease1} | {ease2} |\n\n"
            "Our recommendation: If {condition1}, choose {dest1}. If {condition2}, go with {dest2}."
        )
    },
    {
        "instruction": "Is {dest1} or {dest2} better for a {duration}-day trip on a {budget} budget?",
        "output": (
            "On a {budget} budget for {duration} days, here is how they compare:\n\n"
            "{dest1}: Daily costs in {dest1} average {cost1}. Over {duration} days you would spend roughly "
            "{total1} including accommodation, meals, and activities. {budget_note1}.\n\n"
            "{dest2}: Daily costs average {cost2}, totalling roughly {total2} for {duration} days. {budget_note2}.\n\n"
            "Verdict: {verdict}."
        )
    },
]

STRENGTHS = [
    "cultural heritage and ancient history",
    "natural landscapes and outdoor adventure",
    "food scene and culinary diversity",
    "beach relaxation and water sports",
    "nightlife and urban energy",
    "wildlife and safari experiences",
    "budget-friendly travel",
    "luxury and fine dining",
    "family-friendly activities",
    "romantic ambiance and scenery",
]
DAILY_COSTS = ["$40–60", "$60–80", "$80–120", "$120–180", "$180–250"]
EASE_LEVELS = ["Very easy", "Easy", "Moderate", "Requires planning"]


def gen_comparison(n=150):
    pairs = []
    dest_pairs = [(DESTINATIONS[i], DESTINATIONS[j])
                  for i in range(len(DESTINATIONS))
                  for j in range(i+1, len(DESTINATIONS))]
    random.shuffle(dest_pairs)
    traveler_cycle = TRAVELER_TYPES * (n // len(TRAVELER_TYPES) + 1)
    random.shuffle(traveler_cycle)
    for idx, ((d1, r1, c1), (d2, r2, c2)) in enumerate(dest_pairs[:n]):
        trav = traveler_cycle[idx]
        dur  = random.choice(DURATIONS)
        budget = random.choice(BUDGETS)
        cost1 = random.choice(DAILY_COSTS)
        cost2 = random.choice(DAILY_COSTS)
        s1 = random.choice(STRENGTHS)
        s2 = random.choice([s for s in STRENGTHS if s != s1])
        tmpl = random.choice(COMPARISON_TEMPLATES)
        try:
            instruction = tmpl["instruction"].format(
                dest1=d1, dest2=d2, traveler=trav, duration=dur, budget=budget
            )
            output = tmpl["output"].format(
                dest1=d1, dest2=d2, region1=r1, region2=r2,
                traveler=trav, duration=dur, budget=budget,
                strength1=s1, strength2=s2,
                best_for1=trav + "s seeking " + s1,
                best_for2=trav + "s who love " + s2,
                cost1=cost1, cost2=cost2,
                total1=f"${int(cost1.split('$')[1].split('–')[0]) * dur}–{int(cost1.split('–')[1]) * dur}",
                total2=f"${int(cost2.split('$')[1].split('–')[0]) * dur}–{int(cost2.split('–')[1]) * dur}",
                ease1=random.choice(EASE_LEVELS),
                ease2=random.choice(EASE_LEVELS),
                recommendation=f"As a {trav}, {d1} offers better {s1} while {d2} excels in {s2}",
                condition1=f"you prioritize {s1}",
                condition2=f"{s2} matters more to you",
                budget_note1=f"{d1} is {'budget-friendly' if '$40' in cost1 or '$60' in cost1 else 'mid-to-high range'}",
                budget_note2=f"{d2} is {'budget-friendly' if '$40' in cost2 or '$60' in cost2 else 'mid-to-high range'}",
                verdict=f"{'Both fit' if True else ''} {budget} over {dur} days. {d1} gives more value for {s1}; {d2} wins for {s2}.",
            )
            pairs.append({"instruction": instruction, "output": output, "intent": "destination_comparison"})
        except Exception:
            continue
    return pairs


# ─────────────────────────────────────────────
# Intent 3: Budget Estimation (~180 pairs)
# ─────────────────────────────────────────────

BUDGET_TEMPLATES = [
    {
        "instruction": "Can I visit {country} on a {budget} budget for {days} days?",
        "output": (
            "Yes, {country} is {'very' if budget_tier == 'low' else 'reasonably'} doable on {budget} for {days} days. "
            "Here is how your budget breaks down:\n\n"
            "• Accommodation: {accom_cost} per night ({accom_type})\n"
            "• Meals: {meal_cost} per day ({meal_type})\n"
            "• Local transport: {transport_cost} per day\n"
            "• Activities/entrance fees: {activity_cost} per day\n"
            "• Miscellaneous (tips, SIM card, souvenirs): {misc_cost} total\n\n"
            "Total estimated spend: {total_est}\n\n"
            "Money-saving tips for {country}: {saving_tip}."
        )
    },
    {
        "instruction": "What is the average daily cost of travelling in {country}?",
        "output": (
            "The average daily cost in {country} varies by travel style:\n\n"
            "• Budget traveller: {budget_low}/day — hostels, street food, public transport\n"
            "• Mid-range traveller: {budget_mid}/day — 3-star hotels, restaurant meals, occasional taxis\n"
            "• Luxury traveller: {budget_high}/day — 4–5 star hotels, fine dining, private tours\n\n"
            "Key costs to plan for:\n"
            "- Flights: varies greatly by origin; book 6–8 weeks ahead for best prices\n"
            "- Visa: {visa_note}\n"
            "- Travel insurance: budget $5–10/day — strongly recommended\n\n"
            "Overall, {country} is considered a {value_label} destination for international travellers."
        )
    },
    {
        "instruction": "How much money do I need for a {days}-day trip to {country} as a {traveler}?",
        "output": (
            "For a {traveler} spending {days} days in {country}, here is a realistic budget:\n\n"
            "Daily expenses:\n"
            "• Accommodation: {accom_cost}/night\n"
            "• Food: {meal_cost}/day\n"
            "• Getting around: {transport_cost}/day\n"
            "• Sightseeing & activities: {activity_cost}/day\n"
            "• Daily subtotal: {daily_total}\n\n"
            "Total for {days} days: {trip_total}\n"
            "Plus flights: {flight_note}\n"
            "Emergency buffer (10%): {buffer}\n\n"
            "Grand total to budget: {grand_total}\n\n"
            "As a {traveler}, you can save significantly by {saving_tip}."
        )
    },
]

BUDGET_TIERS = {
    "$500":  ("low",    "$15–25",  "$5–10",   "$5",   "$10",  "$30",  "$450–500"),
    "$1,000": ("low",   "$25–40",  "$10–15",  "$8",   "$15",  "$50",  "$900–1000"),
    "$1,500": ("mid",   "$40–70",  "$15–25",  "$12",  "$20",  "$80",  "$1,300–1,500"),
    "$2,000": ("mid",   "$60–90",  "$20–35",  "$15",  "$30",  "$100", "$1,800–2,000"),
    "$3,000": ("mid",   "$80–120", "$30–50",  "$20",  "$40",  "$150", "$2,700–3,000"),
    "$5,000": ("high",  "$120–200","$50–80",  "$30",  "$60",  "$200", "$4,500–5,000"),
    "$8,000": ("high",  "$200–350","$80–150", "$50",  "$100", "$300", "$7,500–8,000"),
}
SAVING_TIPS = [
    "booking accommodation 3–4 weeks in advance and eating at local markets",
    "using public transport instead of taxis and staying in guesthouses",
    "travelling in shoulder season (April–May or September–October)",
    "buying a local SIM card on arrival instead of roaming",
    "purchasing a city tourist pass for unlimited public transport and museum entry",
    "cooking one meal per day if staying in a hostel with a kitchen",
    "booking free walking tours and self-guided hikes instead of guided tours",
]
VALUE_LABELS = ["very budget-friendly", "affordable", "moderately priced", "mid-range", "premium"]
VISA_NOTES = [
    "many nationalities get visa-on-arrival or e-visa for $20–50",
    "visa-free for most Western passport holders",
    "apply for an e-visa at least 2 weeks before departure",
    "check your country's embassy website for the latest requirements",
]


def gen_budget(n=180):
    pairs = []
    combos = list(product(DESTINATIONS, BUDGETS, DURATIONS, TRAVELER_TYPES))
    random.shuffle(combos)
    for (dest, budget, days, trav) in combos[:n]:
        country, region, cities_str = dest
        tier_data = BUDGET_TIERS.get(budget, BUDGET_TIERS["$2,000"])
        budget_tier, accom, meal, transport, activity, misc, total_est = tier_data
        daily_num = int(accom.split("$")[1].split("–")[0]) + int(meal.split("$")[1].split("–")[0]) + int(transport.replace("$","")) + int(activity.replace("$",""))
        trip_num = daily_num * days
        buffer_num = int(trip_num * 0.1)
        grand_num = trip_num + buffer_num

        tmpl = random.choice(BUDGET_TEMPLATES)
        try:
            instruction = tmpl["instruction"].format(country=country, budget=budget, days=days, traveler=trav)
            output = tmpl["output"].format(
                country=country, budget=budget, days=days, traveler=trav,
                budget_tier=budget_tier,
                accom_cost=accom,
                accom_type="hostel dorms or guesthouses" if budget_tier == "low" else "3-star hotels" if budget_tier == "mid" else "4–5 star hotels",
                meal_cost=meal,
                meal_type="street food and local eateries" if budget_tier == "low" else "mid-range restaurants" if budget_tier == "mid" else "fine dining",
                transport_cost=transport,
                activity_cost=activity,
                misc_cost=misc,
                total_est=total_est,
                budget_low=BUDGET_TIERS["$500"][1],
                budget_mid=BUDGET_TIERS["$2,000"][1],
                budget_high=BUDGET_TIERS["$5,000"][1],
                visa_note=random.choice(VISA_NOTES),
                value_label=random.choice(VALUE_LABELS),
                daily_total=f"${daily_num}/day",
                trip_total=f"${trip_num}",
                flight_note="budget $400–1,200 depending on your origin city",
                buffer=f"${buffer_num}",
                grand_total=f"${grand_num + 800} (including flights)",
                saving_tip=random.choice(SAVING_TIPS),
            )
            pairs.append({"instruction": instruction, "output": output, "intent": "budget_estimation"})
        except Exception:
            continue
    return pairs


# ─────────────────────────────────────────────
# Intent 4: Hotel Booking Help (~150 pairs)
# ─────────────────────────────────────────────

BOOKING_TEMPLATES = [
    {
        "instruction": "What type of accommodation should a {traveler} choose in {country}?",
        "output": (
            "For a {traveler} visiting {country}, here are the best accommodation options:\n\n"
            "1. Hostels / guesthouses ($15–35/night): Great for solo travellers and backpackers. "
            "Social atmosphere, shared facilities, often include breakfast.\n\n"
            "2. Boutique hotels ($60–120/night): Perfect for couples and those wanting a local flavour "
            "without chain-hotel prices. Usually family-run with personal service.\n\n"
            "3. Resort / luxury hotel ($150–400+/night): Ideal for families and honeymooners. "
            "Pools, spas, concierge, and all-inclusive options available.\n\n"
            "Top booking tips:\n"
            "• Book at least 2–3 weeks ahead, especially for peak season in {country}\n"
            "• Read reviews on multiple platforms before booking\n"
            "• Check the cancellation policy — free cancellation options are worth a small premium\n"
            "• Staying near a metro or train station saves time and taxi costs\n\n"
            "For {traveler}s specifically, {specific_tip}."
        )
    },
    {
        "instruction": "How do I find a good hotel in {city} for under {budget}?",
        "output": (
            "Finding a quality hotel in {city} under {budget} is very achievable. Here is how:\n\n"
            "Step 1 — Filter by price: Set your max budget at 10% below {budget} to leave room for taxes.\n\n"
            "Step 2 — Prioritise location: Look for hotels near the city centre, public transport, or the main tourist area. "
            "In {city}, the {area_tip}.\n\n"
            "Step 3 — Check reviews: Focus on properties with 8+ ratings on Booking.com or 4.2+ on Google. "
            "Read the most recent reviews for current conditions.\n\n"
            "Step 4 — Compare platforms: Check Booking.com, Hotels.com, and the hotel's own website — "
            "direct bookings sometimes offer better rates or free breakfast.\n\n"
            "Step 5 — Look for value adds: Free Wi-Fi, airport shuttle, and breakfast can save $20–40/day.\n\n"
            "Pro tip: Mid-week check-ins are typically 10–20% cheaper than weekend arrivals."
        )
    },
    {
        "instruction": "What should I look for when booking a hotel for a {traveler} trip?",
        "output": (
            "When booking a hotel for a {traveler} trip, prioritise these factors:\n\n"
            "Location: Stay within 15 minutes of your main activity zones. "
            "Transport links matter more than the room itself.\n\n"
            "Room type: {room_tip}\n\n"
            "Amenities to check:\n"
            "• Free cancellation (essential for flexible trips)\n"
            "• Breakfast included (saves $10–20/day per person)\n"
            "• Air conditioning and reliable Wi-Fi\n"
            "• 24-hour reception if arriving late\n\n"
            "Red flags to avoid:\n"
            "• Reviews mentioning noise, cleanliness issues, or unresponsive staff\n"
            "• Photos that look professionally staged but reviewers describe differently\n"
            "• No street address listed or vague location descriptions\n\n"
            "Best booking platforms for {traveler}s: {platform_tip}."
        )
    },
]

ROOM_TIPS = {
    "solo traveler": "Single rooms or shared dorms in social hostels work well — great for meeting people",
    "couple": "Ask for a double or queen room. Superior rooms with views are worth the upgrade",
    "family with kids": "Request a family room or interconnecting rooms. Confirm the hotel has a cot if needed",
    "group of friends": "Look for apartments on Airbnb or hotels offering group discounts for 3+ rooms",
    "honeymoon couple": "Splurge on a superior or suite room. Many hotels offer honeymoon packages with extras",
    "senior travelers": "Ground floor or elevator-accessible rooms. Check for grab rails and medical proximity",
    "backpacker": "Dorm beds in well-reviewed hostels with lockers for valuables. Kitchen access is a bonus",
}
PLATFORM_TIPS = {
    "solo traveler": "Hostelworld for dorms, Booking.com for private rooms",
    "couple": "Booking.com or the hotel's website directly for best rates",
    "family with kids": "Hotels.com or Expedia — good filtering for family rooms",
    "group of friends": "Airbnb for whole-apartment bookings, or Booking.com group rates",
    "honeymoon couple": "Mr & Mrs Smith or Small Luxury Hotels for boutique romantic options",
    "senior travelers": "Booking.com with accessibility filters, or AARP Travel for member discounts",
    "backpacker": "Hostelworld, then cross-check on Booking.com for private rooms",
}
AREA_TIPS = [
    "city centre or old town area offers the best walkability",
    "neighbourhood near the main train station reduces transport costs",
    "beach road strip has the most hotel options at varying price points",
    "embassy or business district offers clean, professional hotels",
    "historic quarter gives the most authentic local experience",
]
SPECIFIC_TIPS = {
    "solo traveler": "hostels with common areas and organised events are the best way to meet fellow travellers",
    "couple": "a boutique hotel with a rooftop bar or spa access elevates the experience considerably",
    "family with kids": "resort-style hotels with pools and kids' clubs make the trip stress-free for parents",
    "group of friends": "renting a villa or apartment together is almost always cheaper and more fun than separate hotel rooms",
    "honeymoon couple": "email the hotel before arrival mentioning the occasion — many offer complimentary upgrades or amenities",
    "senior travelers": "call the hotel directly to confirm accessibility features — online listings are not always accurate",
    "backpacker": "read the 'Solo Travellers' tags on hostel reviews — they reveal the social vibe more than star ratings do",
}


def gen_booking(n=150):
    pairs = []
    combos = list(product(DESTINATIONS, TRAVELER_TYPES, BUDGETS))
    random.shuffle(combos)
    for (dest, trav, budget) in combos[:n]:
        country, region, cities_str = dest
        city = cities_str.split(",")[0].strip()
        tmpl = random.choice(BOOKING_TEMPLATES)
        try:
            instruction = tmpl["instruction"].format(country=country, city=city, traveler=trav, budget=budget)
            output = tmpl["output"].format(
                country=country, city=city, traveler=trav, budget=budget,
                specific_tip=SPECIFIC_TIPS.get(trav, "research well and book early"),
                room_tip=ROOM_TIPS.get(trav, "choose a standard double room"),
                platform_tip=PLATFORM_TIPS.get(trav, "Booking.com and Hotels.com"),
                area_tip=random.choice(AREA_TIPS),
            )
            pairs.append({"instruction": instruction, "output": output, "intent": "hotel_booking_help"})
        except Exception:
            continue
    return pairs


# ─────────────────────────────────────────────
# Intent 5: Cancellation / Refund Support (~150 pairs)
# ─────────────────────────────────────────────

CANCEL_SCENARIOS = [
    ("My flight was cancelled. What should I do?",
     "If your flight was cancelled by the airline, you have legal rights in most countries:\n\n"
     "Immediate steps:\n"
     "1. Do NOT leave the airport without speaking to an airline representative first.\n"
     "2. Request a rebooking on the next available flight at no extra charge — this is your right.\n"
     "3. If you prefer, request a full refund to your original payment method.\n"
     "4. Ask for meal vouchers and accommodation if the delay is over 3 hours (EU law mandates this for EU flights).\n\n"
     "If the airline is unresponsive:\n"
     "• Contact your travel insurance provider immediately\n"
     "• File a complaint with the aviation authority in your departure country\n"
     "• Use a credit card chargeback if the airline refuses a refund\n\n"
     "Document everything: take photos of departure boards, keep all receipts for extra expenses — these are reimbursable."),

    ("How do I cancel a hotel booking and get a refund?",
     "To cancel a hotel booking and request a refund:\n\n"
     "Step 1 — Check your booking's cancellation policy. It is listed in your confirmation email under 'Booking Details'. "
     "Policies range from free cancellation up to 24 hours before check-in to non-refundable rates.\n\n"
     "Step 2 — Cancel through the same platform you booked on (Booking.com, Expedia, hotel website). "
     "Log in, find your booking, and select 'Cancel Booking'.\n\n"
     "Step 3 — For refundable bookings, refunds typically process in 5–10 business days to your original payment method.\n\n"
     "Step 4 — If outside the free cancellation window, contact the hotel directly. Many hotels will offer a credit or date change "
     "instead of a cash refund, especially for emergencies.\n\n"
     "Always cancel in writing (email) and keep the confirmation of cancellation for your records."),

    ("I need to cancel my travel package. Will I lose all my money?",
     "Whether you lose money depends on when you cancel and what type of booking you have:\n\n"
     "More than 60 days before travel: Most tour operators offer full refunds minus a small admin fee ($50–150).\n\n"
     "30–60 days before: Typical penalty is 25–50% of the total package cost.\n\n"
     "14–30 days before: Penalty rises to 50–75%.\n\n"
     "Less than 14 days: Most packages are non-refundable at this stage.\n\n"
     "What you can do:\n"
     "1. Check if your travel insurance covers cancellation — 'cancel for any reason' policies offer the most protection\n"
     "2. Ask the operator to defer your booking to a future date rather than cancel outright\n"
     "3. Transfer your booking to another person — many operators allow this for a fee\n\n"
     "Contact your travel agent or tour operator in writing and request their cancellation policy in full before making a decision."),

    ("The hotel charged me for a room I did not use. How do I dispute this?",
     "Disputing an incorrect hotel charge:\n\n"
     "Step 1 — Contact the hotel directly first. Email or call the front desk and reference your booking number. "
     "Explain the charge and request a correction. Most legitimate hotels resolve billing errors quickly.\n\n"
     "Step 2 — If the hotel is unresponsive, escalate to the booking platform (Booking.com, Expedia, etc.). "
     "They have guest protection policies and can intervene on your behalf.\n\n"
     "Step 3 — File a chargeback with your credit card provider if the charge is clearly incorrect "
     "and the hotel refuses to refund. Provide your booking confirmation, cancellation confirmation, "
     "and any email correspondence as evidence.\n\n"
     "Step 4 — Leave an honest review detailing the billing issue. Hotels take review responses seriously "
     "and this often accelerates resolution.\n\n"
     "Timeline: Most disputes resolve within 7–14 business days."),

    ("What is a typical refund policy for travel insurance?",
     "Travel insurance refund policies vary by provider but generally follow this pattern:\n\n"
     "Free look period: Most policies allow cancellation within 10–15 days of purchase for a full refund, "
     "provided you have not made a claim yet.\n\n"
     "After the free look period: Refunds are rarely available unless the trip has been cancelled by the provider "
     "or there is a qualifying life event (death, serious illness).\n\n"
     "What travel insurance typically covers:\n"
     "• Trip cancellation due to illness, injury, or death of a family member\n"
     "• Flight cancellations and significant delays (usually 6+ hours)\n"
     "• Lost or stolen baggage (up to policy limit)\n"
     "• Medical emergencies abroad\n\n"
     "What it typically does NOT cover:\n"
     "• Changing your mind about travelling\n"
     "• Pre-existing medical conditions (unless declared and accepted)\n"
     "• Pandemic-related cancellations (varies by policy)\n\n"
     "Always read the Product Disclosure Statement before purchasing — the fine print determines everything."),

    ("My tour operator went bankrupt. Can I get my money back?",
     "This is a serious situation but there are avenues to recover your money:\n\n"
     "1. Credit card chargeback: If you paid by credit card, contact your bank immediately and request a chargeback. "
     "This is your fastest and most reliable option. Act within 120 days of the charge date.\n\n"
     "2. Travel insurance: If your policy includes 'supplier default' or 'financial failure' cover, file a claim now. "
     "Gather all booking documents and the operator's bankruptcy notice.\n\n"
     "3. ATOL/ABTA protection (UK): If the operator was ATOL or ABTA bonded, contact them directly. "
     "They exist specifically to protect consumers in this scenario.\n\n"
     "4. Airline tickets: If your flights were booked separately with an airline, those tickets remain valid "
     "— the airline bankruptcy does not affect them.\n\n"
     "5. Legal route: Join the creditor list in the bankruptcy proceedings. This is slow and rarely returns full amounts, "
     "but it is worth registering.\n\n"
     "Lesson for future bookings: Always pay by credit card and use ATOL/ABTA-bonded operators for large bookings."),

    ("Can I get a refund if I miss my flight?",
     "In most cases, missing your flight due to your own fault does not entitle you to a full refund, "
     "but here are your options:\n\n"
     "Non-refundable tickets: You will typically lose the base fare. However, airlines may refund taxes and airport fees "
     "automatically — these can amount to $50–200 depending on the route.\n\n"
     "Flexible/refundable tickets: Contact the airline immediately. You can usually be rebooked on the next available flight "
     "for a change fee, or receive a travel credit.\n\n"
     "If you missed the flight due to airline delay on a connecting flight: The airline is responsible for rebooking you "
     "at no cost. Document the delay and request written confirmation.\n\n"
     "Travel insurance: 'Missed departure' cover may apply if you missed the flight due to a qualifying reason "
     "(public transport failure, accident, etc.) — check your policy.\n\n"
     "Act immediately: Call the airline before the flight departs if possible — 'no show' policies are stricter "
     "than same-day rebooking requests."),

    ("How long does a hotel refund take to process?",
     "Hotel refund timelines depend on how and where you booked:\n\n"
     "Direct hotel booking:\n"
     "• Credit card: 5–10 business days after cancellation confirmation\n"
     "• Debit card: Can take up to 10–14 business days\n"
     "• Bank transfer: 3–7 business days in most cases\n\n"
     "Third-party booking platform (Booking.com, Expedia, etc.):\n"
     "• The platform processes the refund first, then your bank processes it\n"
     "• Total time: 7–14 business days is common\n"
     "• Some platforms credit your account wallet first — check before assuming the refund failed\n\n"
     "If refund has not arrived after 14 days:\n"
     "1. Check your cancellation confirmation email for the refund reference number\n"
     "2. Contact the hotel or platform with that reference and request a trace\n"
     "3. Contact your bank to check if a pending credit is held\n"
     "4. Escalate to the platform's customer support team if still unresolved"),
]

CANCEL_VARIATIONS = [
    ("airline cancelled", "What are my rights if my airline cancels my flight?"),
    ("hotel no show", "Will I be charged if I don't show up to my hotel reservation?"),
    ("tour refund", "I cancelled my tour 2 weeks before. Am I entitled to a refund?"),
    ("double charge", "I was charged twice for the same hotel booking. What do I do?"),
    ("travel insurance claim", "How do I make a travel insurance claim for a cancelled trip?"),
    ("name change", "Can I change the name on my flight ticket and is there a fee?"),
    ("date change", "I need to change my hotel check-in date. Will I be charged?"),
]

CANCEL_VARIATION_OUTPUTS = [
    "Under EU261 (EU flights) and similar rules globally, if the airline cancels your flight you are entitled to: "
    "(1) Full refund or rebooking at no charge, (2) Meals and refreshments during long delays, "
    "(3) Hotel accommodation if stranded overnight, (4) Compensation of €250–600 depending on flight distance (EU only). "
    "Always ask the airline for written confirmation and keep all receipts for additional expenses.",

    "Hotel no-show policies vary by rate type. Non-refundable rates: you will be charged for the first night or the full stay. "
    "Flexible rates: usually no charge if you cancel before the stated deadline (often 24–48 hours before check-in). "
    "Best action: always call the hotel if you know you won't arrive — some hotels will waive charges for genuine emergencies "
    "if you contact them promptly rather than simply not showing up.",

    "Cancelling 2 weeks before typically falls in the 50–75% penalty window for most tour operators. "
    "However, every operator has different terms. Check your original booking confirmation for the exact cancellation schedule. "
    "Options: (1) Ask if you can defer to a later date — often allowed for a smaller fee, "
    "(2) Transfer your spot to a friend, (3) Claim through travel insurance if you have cancellation cover.",

    "A double charge is almost always a billing error. Steps to resolve: "
    "(1) Contact the hotel immediately with both charge dates and amounts, "
    "(2) Ask them to reverse the duplicate charge — most resolve this within 24 hours, "
    "(3) If not, initiate a chargeback with your bank for the duplicate amount with your booking confirmation as evidence. "
    "Keep screenshots of both charges on your bank statement.",

    "To make a travel insurance claim: (1) Notify your insurer as soon as possible — most have a 24-hour claims hotline. "
    "(2) Gather documentation: booking confirmation, cancellation notice, receipts for non-recoverable expenses. "
    "(3) Complete the online claims form on the insurer's portal. "
    "(4) Claims are typically assessed within 10–20 business days. "
    "Tip: Never wait — late notification can invalidate some claims.",

    "Name changes on flight tickets depend on the airline and fare type. "
    "Most low-cost carriers (Ryanair, EasyJet, AirAsia) allow name corrections for a fee ($30–150). "
    "Full name changes (transferring the ticket to someone else) are generally not allowed on non-refundable fares. "
    "For major carriers, contact them at least 72 hours before departure. "
    "Always double-check the name matches your passport exactly at the time of booking.",

    "Changing your hotel check-in date depends on your rate type. "
    "Flexible rates: most hotels allow date changes free of charge with 24–48 hours notice. "
    "Non-refundable rates: changes are usually not allowed, but you can call the hotel directly — "
    "many will accommodate a one-time date change if occupancy allows. "
    "Third-party bookings: contact the platform (Booking.com, Expedia) as hotels cannot always modify third-party reservations directly.",
]


def gen_cancellation(n=150):
    pairs = []
    # Main scenarios
    for instruction, output in CANCEL_SCENARIOS:
        pairs.append({"instruction": instruction, "output": output, "intent": "cancellation_refund_support"})

    # Variation scenarios
    for (_, instruction), output in zip(CANCEL_VARIATIONS, CANCEL_VARIATION_OUTPUTS):
        pairs.append({"instruction": instruction, "output": output, "intent": "cancellation_refund_support"})

    # Pad with destination-specific cancellation questions
    destinations_sample = random.sample(DESTINATIONS, min(n - len(pairs), len(DESTINATIONS)))
    templates = [
        ("I need to cancel my hotel in {city}. What is the process?",
         "To cancel a hotel booking in {city}: (1) Log into the platform where you booked. "
         "(2) Find your booking and select 'Cancel'. (3) Check the refund timeline in the cancellation confirmation. "
         "If you booked direct, call the hotel and confirm cancellation by email. "
         "Refunds for eligible bookings typically take 5–10 business days."),
        ("My flight to {country} was delayed by 8 hours. Am I entitled to compensation?",
         "An 8-hour delay is significant and you likely have rights. Under EU261 (if departing from or arriving in the EU on an EU carrier), "
         "you are entitled to meals, refreshments, and compensation of €250–600. "
         "For non-EU flights, compensation depends on the airline's policy and your country's aviation law. "
         "Contact the airline's customer service desk at the airport and request written confirmation of the delay."),
        ("I want to change my travel dates for my trip to {country}. Is there a penalty?",
         "Date change fees for trips to {country} depend on what you have booked:\n\n"
         "Flights: Most airlines charge a change fee of $50–200 plus any fare difference. Flexible fares allow free date changes. "
         "Check your ticket type in your booking confirmation.\n\n"
         "Hotels: Free cancellation rates usually allow date changes at no cost up to 24–48 hours before check-in. "
         "Non-refundable rates typically do not allow changes.\n\n"
         "Tour packages: Change fees are usually 10–25% of the package cost depending on how far in advance you request the change.\n\n"
         "Always contact the provider in writing and get confirmation before assuming a change is processed."),
        ("My accommodation in {city} was nothing like the photos. Can I get a refund?",
         "Yes, misrepresented accommodation is grounds for a legitimate complaint and possible refund. Steps to take:\n\n"
         "1. Document everything immediately: take photos and videos on arrival showing the discrepancies.\n"
         "2. Raise the issue with the property manager or front desk — give them a chance to fix it or offer alternative rooms.\n"
         "3. If unresolved, contact the booking platform (Booking.com, Airbnb, Expedia) and file a dispute with your evidence.\n"
         "4. Most platforms have a misrepresentation policy that entitles you to a full or partial refund.\n"
         "5. If you paid by credit card, a chargeback is your final option if all else fails.\n\n"
         "Act within 24 hours of check-in — delayed complaints are harder to resolve."),
        ("How do I get a refund for a cancelled tour in {country}?",
         "To get a refund for a cancelled tour in {country}:\n\n"
         "1. Check your booking confirmation for the cancellation and refund policy — look for terms like 'free cancellation window'.\n"
         "2. If the tour operator cancelled (not you), you are typically entitled to a full refund within 14 business days.\n"
         "3. Contact the operator by email referencing your booking number and requesting a refund in writing.\n"
         "4. If booked through a platform (GetYourGuide, Viator, Klook), raise the cancellation through their customer portal.\n"
         "5. If the refund is not processed within 14 days, escalate to your credit card company or bank.\n\n"
         "Tip: Always book tours with a clear free-cancellation policy, especially for trips where weather can affect plans."),
    ]
    for dest in destinations_sample:
        country, region, cities_str = dest
        city = cities_str.split(",")[0].strip()
        tmpl = random.choice(templates)
        pairs.append({
            "instruction": tmpl[0].format(city=city, country=country),
            "output": tmpl[1].format(city=city, country=country),
            "intent": "cancellation_refund_support"
        })

    random.shuffle(pairs)
    return pairs[:n]


# ─────────────────────────────────────────────
# Intent 6: Local Customs / Safety Tips (~170 pairs)
# ─────────────────────────────────────────────

CUSTOMS_DATA = {
    "Japan": {
        "customs": ["Remove shoes before entering homes and many restaurants",
                    "Do not tip — it can be considered rude",
                    "Bow slightly when greeting — depth indicates respect level",
                    "Avoid eating or drinking while walking",
                    "Use two hands when giving or receiving business cards or gifts"],
        "safety": ["Japan is one of the safest countries in the world for tourists",
                   "Typhoon season runs June–October — check forecasts",
                   "Be aware of earthquake safety procedures at your hotel",
                   "Tap water is safe to drink everywhere"]
    },
    "Thailand": {
        "customs": ["Dress modestly when visiting temples — cover shoulders and knees",
                    "Never touch anyone's head — it is considered sacred",
                    "The Royal Family is deeply respected — avoid any criticism",
                    "Remove shoes before entering temples and many homes",
                    "Greet with a Wai (press palms together and bow slightly)"],
        "safety": ["Avoid scams around major tourist attractions — book tours through reputable agencies",
                   "Be cautious on motorbike taxis — always wear a helmet",
                   "Drink bottled water — tap water is not safe for tourists",
                   "Keep valuables in hotel safe and be aware of bag snatching in busy markets"]
    },
    "India": {
        "customs": ["Dress conservatively, especially at religious sites",
                    "Remove shoes before entering temples, mosques, and many homes",
                    "Use your right hand for eating and giving/receiving items",
                    "Public displays of affection are generally frowned upon",
                    "Bargaining is expected and normal in markets"],
        "safety": ["Drink only bottled or filtered water",
                   "Be cautious with street food — choose busy stalls with high turnover",
                   "Women travellers should avoid isolated areas at night",
                   "Register with your country's embassy if travelling to remote regions",
                   "Traffic can be chaotic — use registered taxis or apps like Ola/Uber"]
    },
    "Morocco": {
        "customs": ["Dress modestly in medinas and rural areas",
                    "Ask permission before photographing locals",
                    "Accept mint tea when offered — refusing is considered impolite",
                    "Ramadan hours affect restaurant and shop availability — check dates",
                    "Bargaining is part of market culture — start at 30–40% of the asking price"],
        "safety": ["Be firm but polite with persistent vendors — a clear 'no thank you' is sufficient",
                   "Use official guides rather than unofficial touts in major medinas",
                   "Women should carry a scarf to cover up when needed",
                   "Keep digital copies of your passport and emergency contacts"]
    },
    "Italy": {
        "customs": ["Dress codes apply in churches — no shorts or sleeveless tops",
                    "Cappuccino is a morning drink — ordering it after meals marks you as a tourist",
                    "Cover up when entering the Vatican — dress code is strictly enforced",
                    "Greet with 'Buongiorno' (morning) or 'Buonasera' (evening) when entering shops",
                    "Tipping is appreciated but not mandatory — rounding up the bill is common"],
        "safety": ["Pickpockets operate in tourist-heavy areas like the Colosseum and Rome's metro",
                   "Use authorised taxis from official ranks — avoid unmarked vehicles",
                   "Keep a photocopy of your passport separate from the original",
                   "Be aware of the 'coperto' (cover charge) added to restaurant bills — it is legal and normal"]
    },
    "Australia": {
        "customs": ["Australians are casual and informal — first names are used almost immediately",
                    "'Shouting' rounds at the pub is a strong social custom — buy a round for the group",
                    "Punctuality is valued — being late without notice is considered rude",
                    "Tipping is not mandatory but appreciated for exceptional service"],
        "safety": ["Sun protection is essential — UV levels are extreme, especially Oct–March",
                   "Swim between the red-and-yellow flags at patrolled beaches — rips kill",
                   "Australia has venomous snakes and spiders — shake out shoes and check boots",
                   "Tap water is safe to drink throughout the country"]
    },
    "France": {
        "customs": ["Greet shopkeepers with 'Bonjour' when entering — ignoring this is considered rude",
                    "Meals are leisurely affairs — do not ask for the bill immediately after eating",
                    "Dress smartly, especially in Paris — casual sportswear is frowned upon in restaurants",
                    "Kissing on both cheeks ('la bise') is a common greeting between friends",
                    "Tipping is not required but 5–10% is appreciated at restaurants"],
        "safety": ["Be alert to pickpockets around the Eiffel Tower, Louvre, and on the Metro",
                   "Strikes (grèves) can affect transport with little notice — check news during your trip",
                   "Tap water is safe to drink — ask for 'une carafe d'eau' for free tap water at restaurants",
                   "Emergency number in France: 112 (EU-wide)"]
    },
    "Mexico": {
        "customs": ["Greet with a handshake or cheek kiss depending on the region",
                    "Punctuality is flexible socially but expected in business settings",
                    "Tipping is expected — 10–15% at restaurants is standard",
                    "Dia de los Muertos (Nov 1–2) is a celebrated holiday, not a morbid event",
                    "Bargaining is common at markets but not in established shops or restaurants"],
        "safety": ["Check travel advisories for specific states — safety varies significantly by region",
                   "Use authorised taxis or ride-share apps (Uber, DiDi) rather than street cabs",
                   "Drink bottled water — tap water is not safe even for brushing teeth in some areas",
                   "Do not flash expensive jewellery or electronics in public areas"]
    },
}

GENERIC_CUSTOMS_TEMPLATES = [
    ("What are the local customs I should know before visiting {country}?",
     "Here are the key customs and etiquette rules to know before visiting {country}:\n\n"
     "{customs_list}\n\n"
     "Showing basic awareness of local customs goes a long way. Locals genuinely appreciate when visitors make the effort."),
    ("Is {country} safe for a {traveler}?",
     "{country} is generally safe for {traveler}s when standard precautions are taken. "
     "Key safety points:\n\n{safety_list}\n\n"
     "As with any destination, stay aware of your surroundings, keep copies of important documents, "
     "and register with your country's embassy for extended stays."),
    ("What should I know about safety and etiquette in {country}?",
     "Combining safety and etiquette for {country}:\n\n"
     "Cultural etiquette:\n{customs_list}\n\n"
     "Safety tips:\n{safety_list}\n\n"
     "Being culturally respectful and situationally aware will make your trip significantly more enjoyable."),
]


def gen_customs(n=170):
    pairs = []
    countries_with_data = list(CUSTOMS_DATA.keys())

    for country, data in CUSTOMS_DATA.items():
        customs_list = "\n".join(f"• {c}" for c in data["customs"])
        safety_list  = "\n".join(f"• {s}" for s in data["safety"])
        for tmpl in GENERIC_CUSTOMS_TEMPLATES:
            for trav in random.sample(TRAVELER_TYPES, 2):
                instruction = tmpl[0].format(country=country, traveler=trav)
                output = tmpl[1].format(
                    country=country, traveler=trav,
                    customs_list=customs_list, safety_list=safety_list
                )
                pairs.append({"instruction": instruction, "output": output, "intent": "local_customs_safety"})

    # Fill remaining with generic destination pairs
    generic_tmpl = [
        ("Are there any dress code rules I should follow in {country}?",
         "Dress code expectations in {country} depend on the context:\n\n"
         "Religious sites: Modest dress is required — cover shoulders and knees for both men and women. "
         "Scarves or sarongs (often available at the entrance) can be used if you are not dressed appropriately.\n\n"
         "City and everyday settings: Smart casual is acceptable in most urban areas. "
         "Avoid overly revealing clothing in conservative regions.\n\n"
         "Beach areas: Swimwear is fine at the beach and pool, but cover up when walking to restaurants or markets.\n\n"
         "When in doubt, dress more conservatively — it is always easier to remove a layer than to be refused entry."),
        ("What emergency numbers should I know when travelling in {country}?",
         "Key emergency contacts for {country}:\n\n"
         "• Police: 112 (international emergency, works in most countries) or the local number\n"
         "• Medical emergency: 112 or local ambulance number\n"
         "• Your country's embassy: Save this number before departure\n"
         "• Travel insurance 24-hour helpline: From your policy documents\n\n"
         "Preparation tips:\n"
         "• Save all numbers offline — you may not have internet in an emergency\n"
         "• Carry a card with your blood type, allergies, and emergency contact in English and the local language\n"
         "• Know the address of your accommodation in the local script for taxis and emergencies"),
    ]
    remaining = n - len(pairs)
    dest_sample = random.choices(DESTINATIONS, k=remaining)
    for dest in dest_sample:
        country, region, cities_str = dest
        tmpl = random.choice(generic_tmpl)
        pairs.append({
            "instruction": tmpl[0].format(country=country),
            "output": tmpl[1].format(country=country),
            "intent": "local_customs_safety"
        })

    random.shuffle(pairs)
    return pairs[:n]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    existing = sum(1 for _ in open(OUTPUT_FILE, encoding="utf-8"))
    print(f"Existing pairs in {OUTPUT_FILE}: {existing}")

    all_new = []

    print("\nGenerating AI pairs...")
    p1 = gen_itinerary(200);    print(f"  itinerary_planning:          {len(p1)} pairs")
    p2 = gen_comparison(150);   print(f"  destination_comparison:      {len(p2)} pairs")
    p3 = gen_budget(180);       print(f"  budget_estimation:           {len(p3)} pairs")
    p4 = gen_booking(150);      print(f"  hotel_booking_help:          {len(p4)} pairs")
    p5 = gen_cancellation(150); print(f"  cancellation_refund_support: {len(p5)} pairs")
    p6 = gen_customs(170);      print(f"  local_customs_safety:        {len(p6)} pairs")

    all_new = p1 + p2 + p3 + p4 + p5 + p6
    random.shuffle(all_new)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for item in all_new:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = existing + len(all_new)

    print("\n" + "═" * 52)
    print("GENERATION COMPLETE")
    print("═" * 52)

    from collections import Counter
    # Count all intents in the full file
    all_intents = Counter()
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                all_intents[json.loads(line)["intent"]] += 1
            except Exception:
                pass

    for intent, count in sorted(all_intents.items()):
        bar = "█" * (count // 20)
        print(f"  {intent:<35} {count:>4}  {bar}")

    print(f"\n  New pairs added : {len(all_new)}")
    print(f"  Total pairs     : {total}")
    print(f"  Output file     : {OUTPUT_FILE}")
    print("═" * 52)


if __name__ == "__main__":
    main()
