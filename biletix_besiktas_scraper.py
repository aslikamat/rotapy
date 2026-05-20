"""
Biletix — Beşiktaş Events Scraper
===================================
Scrapes only events held in the Beşiktaş district of Istanbul
where the venue capacity is greater than 500.

How capacity is determined (in order):
  1. Known venue database (hardcoded, most reliable)
  2. Scraped from the Biletix venue detail page
  3. Skipped if capacity cannot be determined

Requirements:
    pip install requests beautifulsoup4 playwright pandas
    playwright install chromium

Usage:
    python biletix_besiktas_scraper.py              # scrape + print results
    python biletix_besiktas_scraper.py --csv        # also save CSV
    python biletix_besiktas_scraper.py --min 1000   # change capacity threshold
    python biletix_besiktas_scraper.py --demo       # run with sample data
"""

import re
import time
import json
import argparse
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# KNOWN VENUE DATABASE  (Beşiktaş district, Istanbul)
# Capacity sourced from official venue/Wikipedia data.
# Add more venues here as needed.
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_VENUES = {
    # ── Large stadiums ──────────────────────────────────────────────────────
    "tüpraş stadyumu":          42684,
    "vodafone park":            42684,
    "vodafone arena":           42684,
    "beşiktaş stadyumu":        42684,
    "bjk stadyumu":             42684,

    # ── Concert / multi-purpose halls ───────────────────────────────────────
    "zorlu psm":                2350,   # main hall
    "zorlu center":             2350,
    "zorlu performans sanatları merkezi": 2350,
    "çırağan palace":           700,
    "çırağan sarayı":           700,
    "swissotel the bosphorus":  800,
    "swissôtel":                800,
    "the bosphorus ballroom":   600,
    "beşiktaş kültür merkezi":  800,
    "bkm sahne":                580,
    "bkm":                      580,
    "harbiye cemil topuzlu açık hava tiyatrosu": 5000,
    "harbiye açıkhava":         5000,
    "cemil topuzlu açık hava":  5000,
    "dj room beşiktaş":         200,    # below threshold → filtered out
    "arkaoda":                  150,    # below threshold → filtered out

    # ── Hotels / event spaces ────────────────────────────────────────────────
    "four seasons istanbul at the bosphorus": 600,
    "radisson blu bosphorus":   550,
    "the ritz-carlton istanbul": 600,
    "istanbul kongre merkezi":  3500,   # technically Harbiye/Şişli border
    "lütfi kırdar kongre merkezi": 3500,
}


def lookup_capacity(venue_name: str) -> int | None:
    """
    Try to find venue capacity from the known database.
    Normalises the venue name before matching.
    Returns capacity (int) or None if unknown.
    """
    normalised = venue_name.lower().strip()
    # Exact match first
    if normalised in KNOWN_VENUES:
        return KNOWN_VENUES[normalised]
    # Partial match (venue name contains a known key, or vice versa)
    for key, cap in KNOWN_VENUES.items():
        if key in normalised or normalised in key:
            return cap
    return None


def is_besiktas(venue_name: str, district_text: str = "") -> bool:
    """
    Heuristic: is this venue in the Beşiktaş district?
    Checks venue name + any additional district/address text.
    """
    besiktas_keywords = [
        "beşiktaş", "besiktas",
        "zorlu",        # Zorlu PSM is in Beşiktaş
        "çırağan",      # Çırağan Palace — Beşiktaş waterfront
        "ciragan",
        "swissotel",    # Swissôtel Beşiktaş
        "levent",       # technically Beşiktaş administrative boundary
        "balmumcu",
        "dikilitaş",
        "akaretler",
        "ihlamur",
        "yıldız",
        "harbiye",      # debatable — often included in Beşiktaş searches
        "istanbul kongre", # Lütfi Kırdar — Harbiye
    ]
    combined = (venue_name + " " + district_text).lower()
    return any(kw in combined for kw in besiktas_keywords)


# ──────────────────────────────────────────────────────────────────────────────
# SCRAPER
# ──────────────────────────────────────────────────────────────────────────────

def scrape_biletix_besiktas(min_capacity: int = 500):
    """
    Main entry point.
    Returns a list of event dicts that:
      - are held at a venue in the Beşiktaş district
      - have a known/scraped capacity > min_capacity
    """
    print(f"\n🎭 Biletix Beşiktaş scraper — min capacity: {min_capacity}\n")

    try:
        from playwright.sync_api import sync_playwright
        all_events = _scrape_playwright()
    except Exception as e:
        print(f"⚠️  Playwright failed ({e}), falling back to requests...")
        all_events = _scrape_requests()

    print(f"\n📋 Total raw events collected: {len(all_events)}")

    filtered = filter_events(all_events, min_capacity)

    print(f"✅ Events matching criteria (Beşiktaş + capacity > {min_capacity}): {len(filtered)}\n")
    return filtered


def _scrape_playwright():
    """Playwright (headless Chromium) scraper — handles JS-rendered pages."""
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup

    events = []

    # Biletix category URLs — we scrape all, then filter
    categories = {
        "Konser":   "https://www.biletix.com/kategori/KONSER/TURKIYE/tr",
        "Tiyatro":  "https://www.biletix.com/kategori/TIYATRO/TURKIYE/tr",
        "Spor":     "https://www.biletix.com/kategori/SPOR/TURKIYE/tr",
        "Festival": "https://www.biletix.com/kategori/FESTIVAL/TURKIYE/tr",
        "Sanat":    "https://www.biletix.com/kategori/SANAT/TURKIYE/tr",
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            locale="tr-TR",
        )
        page = ctx.new_page()

        # Warm-up — get cookies
        print("  → Connecting to biletix.com...")
        page.goto("https://www.biletix.com/anasayfa/TURKIYE/tr", timeout=30000)
        page.wait_for_load_state("networkidle", timeout=20000)

        for cat_name, url in categories.items():
            print(f"  → Scraping: {cat_name}")
            try:
                page.goto(url, timeout=20000)
                page.wait_for_load_state("networkidle", timeout=15000)

                # Scroll to load lazy items
                for _ in range(4):
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    time.sleep(0.7)

                soup = BeautifulSoup(page.content(), "html.parser")
                found = _parse_event_cards(soup, cat_name)
                events.extend(found)
                print(f"     ✓ {len(found)} events found")

            except Exception as exc:
                print(f"     ✗ Failed: {exc}")

        # ── For each event that might be in Beşiktaş, visit its detail page ──
        # to get a more precise venue name / district / capacity hint
        besiktas_candidates = [e for e in events if is_besiktas(e["venue"])]
        print(f"\n  → Visiting detail pages for {len(besiktas_candidates)} candidate events...")

        for event in besiktas_candidates:
            if not event.get("url"):
                continue
            try:
                page.goto(event["url"], timeout=15000)
                page.wait_for_load_state("domcontentloaded", timeout=10000)
                detail_soup = BeautifulSoup(page.content(), "html.parser")
                _enrich_from_detail(event, detail_soup)
                time.sleep(0.5)
            except Exception:
                pass  # keep the event with whatever we have

        browser.close()

    return events


def _scrape_requests():
    """Fallback: requests + BeautifulSoup."""
    import requests
    from bs4 import BeautifulSoup

    events = []
    session = requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "tr-TR,tr;q=0.9",
    }

    categories = {
        "Konser":   "https://www.biletix.com/kategori/KONSER/TURKIYE/tr",
        "Tiyatro":  "https://www.biletix.com/kategori/TIYATRO/TURKIYE/tr",
        "Spor":     "https://www.biletix.com/kategori/SPOR/TURKIYE/tr",
        "Festival": "https://www.biletix.com/kategori/FESTIVAL/TURKIYE/tr",
    }

    # Warm-up
    session.get("https://www.biletix.com/anasayfa/TURKIYE/tr", headers=headers, timeout=15)

    for cat_name, url in categories.items():
        print(f"  → Scraping: {cat_name}")
        try:
            r = session.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            found = _parse_event_cards(soup, cat_name)
            events.extend(found)
            print(f"     ✓ {len(found)} events")
            time.sleep(1.2)
        except Exception as exc:
            print(f"     ✗ {exc}")

    return events


def _parse_event_cards(soup, category: str) -> list:
    """
    Parse event cards from a Biletix listing page.
    Tries multiple CSS selectors because Biletix redesigns periodically.
    """
    events = []

    # Multiple selector attempts (most → least specific)
    selectors = [
        "div.event-card",
        "div.eventCard",
        "li.event-item",
        "article.event",
        "div[class*='EventCard']",
        "div[class*='event-card']",
        "a[href*='/etkinlik/']",
        "a[href*='/event/']",
    ]

    cards = []
    for sel in selectors:
        cards = soup.select(sel)
        if cards:
            break

    for card in cards:
        try:
            name = (
                _text(card, "h2") or
                _text(card, "h3") or
                _text(card, "[class*='title']") or
                _text(card, "[class*='name']") or
                card.get_text(strip=True)[:80]
            )
            if not name or len(name) < 3:
                continue

            date_str = (
                _text(card, "[class*='date']") or
                _text(card, "time") or
                _text(card, "[class*='Date']") or
                "Tarih belirtilmemiş"
            )

            venue = (
                _text(card, "[class*='venue']") or
                _text(card, "[class*='location']") or
                _text(card, "[class*='Venue']") or
                _text(card, "[class*='mekan']") or
                "Mekan belirtilmemiş"
            )

            price = (
                _text(card, "[class*='price']") or
                _text(card, "[class*='Price']") or
                _text(card, "[class*='fiyat']") or
                "Belirtilmemiş"
            )

            link = card if card.name == "a" else card.find("a")
            url = ""
            if link and link.get("href"):
                href = link["href"]
                url = href if href.startswith("http") else f"https://www.biletix.com{href}"

            events.append({
                "name":     name.strip(),
                "category": category,
                "date":     date_str.strip(),
                "venue":    venue.strip(),
                "price":    price.strip(),
                "url":      url,
                "district": "",      # filled by detail scrape
                "capacity": None,    # filled by filter_events
            })

        except Exception:
            continue

    return events


def _enrich_from_detail(event: dict, soup):
    """
    Visit the event detail page and try to extract:
      - More precise venue name
      - District / address (to confirm Beşiktaş)
      - Any capacity mention
    """
    # Try to get venue from detail page (usually more complete)
    venue_detail = (
        _text(soup, "[class*='venue']") or
        _text(soup, "[class*='mekan']") or
        _text(soup, "[class*='location']") or
        ""
    )
    if venue_detail and len(venue_detail) > len(event["venue"]):
        event["venue"] = venue_detail.strip()

    # District / address
    address = (
        _text(soup, "[class*='address']") or
        _text(soup, "[class*='adres']") or
        _text(soup, "[class*='district']") or
        _text(soup, "[class*='ilce']") or
        ""
    )
    event["district"] = address.strip()

    # Look for capacity in page text (sometimes listed in venue info)
    page_text = soup.get_text(" ", strip=True)
    cap_match = re.search(
        r"kapasit[ei].*?(\d[\d.,]+)\s*(kişi|seat|koltuk)?",
        page_text, re.IGNORECASE
    )
    if cap_match:
        raw = cap_match.group(1).replace(".", "").replace(",", "")
        try:
            event["_scraped_capacity"] = int(raw)
        except ValueError:
            pass


def _text(element, selector: str) -> str:
    found = element.select_one(selector)
    return found.get_text(strip=True) if found else ""


# ──────────────────────────────────────────────────────────────────────────────
# FILTER
# ──────────────────────────────────────────────────────────────────────────────

def filter_events(events: list, min_capacity: int = 500) -> list:
    """
    Filter events to only those in Beşiktaş with capacity > min_capacity.
    Adds 'capacity' and 'capacity_source' fields to each kept event.
    """
    results = []

    for event in events:
        venue = event.get("venue", "")
        district = event.get("district", "")

        # ── Step 1: must be in Beşiktaş ──────────────────────────────────────
        if not is_besiktas(venue, district):
            continue

        # ── Step 2: determine capacity ────────────────────────────────────────
        capacity = lookup_capacity(venue)
        source = "database"

        if capacity is None and event.get("_scraped_capacity"):
            capacity = event["_scraped_capacity"]
            source = "scraped"

        if capacity is None:
            # Can't determine — skip to be safe
            # Change to: capacity = 0  if you want to include unknowns
            print(f"  ⚠ Skipping (capacity unknown): {event['name']} @ {venue}")
            continue

        # ── Step 3: apply capacity threshold ─────────────────────────────────
        if capacity <= min_capacity:
            print(f"  ✗ Too small ({capacity}): {event['name']} @ {venue}")
            continue

        event["capacity"] = capacity
        event["capacity_source"] = source
        event.pop("_scraped_capacity", None)
        results.append(event)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def print_results(events: list):
    if not events:
        print("❌ No events found matching the criteria.")
        return

    print("=" * 70)
    print(f"  BEŞIKTAŞ EVENTS — capacity > threshold")
    print("=" * 70)

    for i, e in enumerate(events, 1):
        print(f"\n{i}. {e['name']}")
        print(f"   📅 {e['date']}")
        print(f"   🏛️  {e['venue']}")
        print(f"   👥 Kapasite: {e['capacity']:,} ({e['capacity_source']})")
        print(f"   🎟️  {e['category']} — {e['price']}")
        if e.get("url"):
            print(f"   🔗 {e['url']}")

    print("\n" + "=" * 70)
    print(f"  Toplam: {len(events)} etkinlik")
    print("=" * 70)


def save_csv(events: list, path: str = "besiktas_events.csv"):
    if not events:
        print("No data to save.")
        return
    try:
        import pandas as pd
        df = pd.DataFrame(events)
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except ImportError:
        import csv
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=events[0].keys())
            writer.writeheader()
            writer.writerows(events)
    print(f"\n💾 CSV saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# DEMO DATA  (for testing without internet)
# ──────────────────────────────────────────────────────────────────────────────

DEMO_EVENTS_RAW = [
    {"name": "Sertab Erener Konseri",     "category": "Konser",  "date": "15 Mayıs 2025", "venue": "Zorlu PSM",            "price": "350 TL", "url": "https://biletix.com", "district": "Beşiktaş", "capacity": None},
    {"name": "Galatasaray - Fenerbahçe",  "category": "Spor",    "date": "25 Mayıs 2025", "venue": "Tüpraş Stadyumu",       "price": "600 TL", "url": "https://biletix.com", "district": "Beşiktaş", "capacity": None},
    {"name": "Mor ve Ötesi",              "category": "Konser",  "date": "22 Mayıs 2025", "venue": "KüçükÇiftlik Park",     "price": "400 TL", "url": "https://biletix.com", "district": "Şişli",    "capacity": None},
    {"name": "Küçük Sahne Gösterisi",     "category": "Tiyatro", "date": "20 Mayıs 2025", "venue": "Arkaoda Beşiktaş",      "price": "150 TL", "url": "https://biletix.com", "district": "Beşiktaş", "capacity": None},
    {"name": "Harbiye Açıkhava Konseri",  "category": "Konser",  "date": "1 Haziran 2025", "venue": "Harbiye Açıkhava",     "price": "500 TL", "url": "https://biletix.com", "district": "Harbiye",  "capacity": None},
    {"name": "BKM Sahne Gösterisi",       "category": "Tiyatro", "date": "18 Mayıs 2025", "venue": "BKM Sahne",             "price": "200 TL", "url": "https://biletix.com", "district": "Beşiktaş", "capacity": None},
    {"name": "Çırağan Sarayı Galası",     "category": "Sanat",   "date": "28 Mayıs 2025", "venue": "Çırağan Sarayı",       "price": "800 TL", "url": "https://biletix.com", "district": "Beşiktaş", "capacity": None},
    {"name": "Zorlu PSM Festival",        "category": "Festival","date": "5 Haziran 2025","venue": "Zorlu Performans Sanatları Merkezi", "price": "300 TL", "url": "https://biletix.com", "district": "Beşiktaş", "capacity": None},
]


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Biletix Beşiktaş scraper — capacity-filtered events"
    )
    parser.add_argument("--min",  type=int, default=500,
                        help="Minimum venue capacity (default: 500)")
    parser.add_argument("--csv",  action="store_true",
                        help="Save results to CSV")
    parser.add_argument("--demo", action="store_true",
                        help="Use sample data (no internet needed)")
    parser.add_argument("--json", action="store_true",
                        help="Print results as JSON")
    args = parser.parse_args()

    if args.demo:
        print("🎭 Demo mode — using sample data\n")
        filtered = filter_events(DEMO_EVENTS_RAW, args.min)
    else:
        filtered = scrape_biletix_besiktas(args.min)

    if args.json:
        print(json.dumps(filtered, ensure_ascii=False, indent=2))
    else:
        print_results(filtered)

    if args.csv:
        save_csv(filtered, "besiktas_events.csv")

    return filtered


if __name__ == "__main__":
    main()
