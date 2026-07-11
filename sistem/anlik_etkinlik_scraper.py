"""
Anlık Etkinlik Scraper — Model İnference İçin
==============================================
Bugün ve yakın günlerdeki Beşiktaş etkinliklerini çeker.
LSTM modelinin tahmin aşamasında "şu an etkinlik var mı?" sorusunu
yanıtlamak için kullanılır.

Kaynaklar:
  1. Radar Türkiye  — anlık, 16 platform birleşik
  2. Ticketmaster   — API ile (--tm-key)

Kullanım:
  python anlik_etkinlik_scraper.py                  # bugün + 7 gün
  python anlik_etkinlik_scraper.py --days 1         # sadece bugün
  python anlik_etkinlik_scraper.py --tm-key KEY     # + Ticketmaster
  python anlik_etkinlik_scraper.py --json           # JSON çıktı (API için)
  python anlik_etkinlik_scraper.py --check          # şu an etkinlik var mı?

Model entegrasyonu:
  from anlik_etkinlik_scraper import get_active_events, is_event_now
  events = get_active_events()          # liste döner
  flag   = is_event_now(lat, lon)       # 0 veya 1 döner (LSTM feature)

────────────────────────────────────────────────────────────────────
DEĞİŞİKLİK GEÇMİŞİ
────────────────────────────────────────────────────────────────────
- `event_date` + `start_time` ve `event_date` + `end_time` birleştirildi:
  `start_date_hour` / `end_date_hour` sütunları (dakika bilgisi YOK,
  örn. "2026-07-11 20"). Dakika hassasiyeti gereken aktiflik kontrolü
  için `start_datetime` / `end_datetime` (tam ISO) korunuyor.
- YENİ: her etkinliğin mekân koordinatından (lat/lon) üretilen
  `geohash` sütunu eklendi — trafik ve hava durumu scriptleriyle
  aynı algoritma/hassasiyet, birleştirme (join) için kullanılabilir.
"""

import re
import time
import json
import argparse
from datetime import datetime, timedelta

import requests
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# MEKAN VERİTABANI
# ─────────────────────────────────────────────────────────────────────────────

VENUE_DB = {
    "vodafone park":                {"lat": 41.0390, "lon": 29.0100, "cap": 42684},
    "tüpraş stadyumu":              {"lat": 41.0390, "lon": 29.0100, "cap": 42684},
    "beşiktaş tüpraş stadyumu":     {"lat": 41.0390, "lon": 29.0100, "cap": 42684},
    "zorlu psm":                    {"lat": 41.0681, "lon": 29.0116, "cap": 2350},
    "zorlu performans sanatları":   {"lat": 41.0681, "lon": 29.0116, "cap": 2350},
    "harbiye cemil topuzlu":        {"lat": 41.0487, "lon": 28.9887, "cap": 5000},
    "ibb harbiye":                  {"lat": 41.0487, "lon": 28.9887, "cap": 5000},
    "if performance hall beşiktaş": {"lat": 41.0435, "lon": 29.0068, "cap": 800},
    "if performance hall":          {"lat": 41.0435, "lon": 29.0068, "cap": 800},
    "sahne beşiktaş":               {"lat": 41.0428, "lon": 29.0050, "cap": 500},
    "çarşı pub":                    {"lat": 41.0410, "lon": 29.0045, "cap": 300},
    "mentalist pub beşiktaş":       {"lat": 41.0415, "lon": 29.0048, "cap": 250},
    "default_besiktas":             {"lat": 41.0402, "lon": 29.0097, "cap": 500},
}

BESIKTAS_KW = [
    "beşiktaş", "besiktas", "vodafone", "tüpraş", "tupras",
    "zorlu", "çırağan", "harbiye", "ortaköy", "if performance",
    "sahne beşiktaş", "balmumcu", "bebek", "akaretler",
]

def venue_info(name: str) -> dict:
    n = name.lower().strip()
    for key, data in VENUE_DB.items():
        if key in n or n in key:
            return data
    return VENUE_DB["default_besiktas"]

def is_besiktas(venue: str, extra: str = "") -> bool:
    combined = (venue + " " + extra).lower()
    return any(kw in combined for kw in BESIKTAS_KW)


# ─────────────────────────────────────────────────────────────────────────────
# GEOHASH — harici kütüphane gerektirmeyen saf Python implementasyonu
# (trafik ve hava durumu scriptleriyle AYNI algoritma/hassasiyet)
# ─────────────────────────────────────────────────────────────────────────────

GEOHASH_PRECISION = 7
_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def encode_geohash(lat: float, lon: float, precision: int = GEOHASH_PRECISION) -> str:
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon > mid:
                ch |= bits[bit]
                lon_range[0] = mid
            else:
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat > mid:
                ch |= bits[bit]
                lat_range[0] = mid
            else:
                lat_range[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(_GEOHASH_BASE32[ch])
            bit = 0
            ch = 0
    return "".join(geohash)


# ─────────────────────────────────────────────────────────────────────────────
# RADAR TÜRKİYE — anlık çekim
# ─────────────────────────────────────────────────────────────────────────────

RADAR_BASE = "https://www.radarturkiye.com"

TR_MONTHS = {
    "ocak": 1, "şubat": 2, "mart": 3, "nisan": 4,
    "mayıs": 5, "haziran": 6, "temmuz": 7, "ağustos": 8,
    "eylül": 9, "ekim": 10, "kasım": 11, "aralık": 12,
}

def _parse_tr_date(text: str) -> str:
    m = re.search(
        r'(\d{1,2})\s+([A-Za-zğışüöçĞİŞÜÖÇ]+)\s*(\d{4})?', text
    )
    if m:
        day  = int(m.group(1))
        mon  = TR_MONTHS.get(m.group(2).lower())
        year = int(m.group(3)) if m.group(3) else datetime.now().year
        if mon:
            return f"{year}-{mon:02d}-{day:02d}"
    m2 = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
    return m2.group(0) if m2 else ""

def _parse_time(text: str) -> str:
    m = re.search(r'\b(\d{1,2}):(\d{2})\b', text)
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else ""

def _guess_category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["konser", "concert"]):  return "Konser"
    if any(k in n for k in ["tiyatro", "oyun"]):    return "Tiyatro"
    if any(k in n for k in ["stand-up", "komedi"]): return "Stand-up"
    if any(k in n for k in ["maç", "futbol"]):      return "Spor"
    if any(k in n for k in ["festival"]):           return "Festival"
    return "Etkinlik"

def _estimate_end(start_dt: datetime, category: str) -> datetime:
    hours = {"Spor": 2.5, "Konser": 3.0, "Festival": 4.0,
             "Tiyatro": 2.5, "Stand-up": 2.0}.get(category, 2.5)
    return start_dt + timedelta(hours=hours)


def _date_hour(dt: datetime | None, fallback_date: str = "") -> str:
    """
    Tarih + saati tek sütunda birleştirir, dakika bilgisi YOK.
    dt varsa  -> "YYYY-MM-DD HH"
    dt yoksa  -> sadece fallback_date (saat bilinmiyor)
    """
    if dt:
        return dt.strftime("%Y-%m-%d %H")
    return fallback_date


def scrape_radar_anlik(days_ahead: int = 7) -> list[dict]:
    """
    Radar Türkiye'den bugün + days_ahead gün içindeki
    Beşiktaş etkinliklerini çeker.
    """
    print("[Radar Türkiye] Anlık etkinlikler çekiliyor...")

    try:
        from playwright.sync_api import sync_playwright
        from bs4 import BeautifulSoup
        return _radar_pw_anlik(days_ahead)
    except Exception:
        return _radar_req_anlik(days_ahead)


def _radar_pw_anlik(days_ahead: int) -> list[dict]:
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup

    UA = "Mozilla/5.0 Chrome/123.0.0.0 Safari/537.36"
    today    = datetime.now()
    end_date = today + timedelta(days=days_ahead)
    events   = []

    search_terms = [
        "beşiktaş", "vodafone park", "zorlu psm",
        "if performance hall beşiktaş", "harbiye",
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_context(user_agent=UA).new_page()

        for term in search_terms:
            url = f"{RADAR_BASE}/ara?q={requests.utils.quote(term)}"
            try:
                page.goto(url, timeout=20000)
                page.wait_for_load_state("networkidle", timeout=15000)
                for _ in range(3):
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    time.sleep(0.5)

                soup  = BeautifulSoup(page.content(), "html.parser")
                found = _parse_radar_cards_anlik(soup, today, end_date)
                events.extend(found)
                print(f"  → '{term}': {len(found)} etkinlik")
            except Exception as e:
                print(f"  → '{term}': [HATA] {e}")
            time.sleep(0.8)

        browser.close()
    return events


def _radar_req_anlik(days_ahead: int) -> list[dict]:
    from bs4 import BeautifulSoup

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 Chrome/123.0.0.0 Safari/537.36"

    today    = datetime.now()
    end_date = today + timedelta(days=days_ahead)
    events   = []

    for term in ["beşiktaş", "vodafone park", "zorlu psm", "harbiye"]:
        url = f"{RADAR_BASE}/ara?q={requests.utils.quote(term)}"
        try:
            r    = session.get(url, timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")
            found = _parse_radar_cards_anlik(soup, today, end_date)
            events.extend(found)
            print(f"  → '{term}': {len(found)} etkinlik")
            time.sleep(1)
        except Exception as e:
            print(f"  → '{term}': [HATA] {e}")
    return events


def _parse_radar_cards_anlik(soup, today: datetime, end_date: datetime) -> list[dict]:
    events = []
    selectors = [
        "article", "div.event-card", "div[class*='EventCard']",
        "div[class*='event']", "a[href*='/etkinlik/']",
    ]
    cards = []
    for sel in selectors:
        cards = soup.select(sel)
        if len(cards) > 2:
            break

    for card in cards:
        try:
            # Ad
            name = ""
            for tag in ["h1","h2","h3","[class*='title']","[class*='name']"]:
                el = card.select_one(tag)
                if el:
                    name = el.get_text(strip=True)
                    break
            if not name or len(name) < 3:
                continue

            # Tarih + saat
            full_text = card.get_text(" ", strip=True)
            date_str  = _parse_tr_date(full_text)
            time_str  = _parse_time(full_text)

            if not date_str:
                continue

            # Tarih filtresi
            try:
                ev_date = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                continue
            if not (today.date() <= ev_date.date() <= end_date.date()):
                continue

            # Mekan
            venue = ""
            for tag in ["[class*='venue']","[class*='mekan']","[class*='location']"]:
                el = card.select_one(tag)
                if el:
                    venue = el.get_text(strip=True)
                    break

            if not is_besiktas(venue, name):
                continue

            # URL
            link = card if card.name == "a" else card.find("a")
            url  = ""
            if link and link.get("href"):
                h = link["href"]
                url = h if h.startswith("http") else f"{RADAR_BASE}{h}"

            category = _guess_category(name)
            vi       = venue_info(venue)

            start_dt = None
            end_dt   = None
            if time_str:
                try:
                    start_dt = datetime.strptime(
                        f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
                    )
                    end_dt = _estimate_end(start_dt, category)
                except Exception:
                    pass

            events.append({
                "source":               "Radar Türkiye",
                "name":                 name,
                "category":             category,
                "start_date_hour":      _date_hour(start_dt, date_str),
                "end_date_hour":        _date_hour(end_dt, ""),
                "start_datetime":       start_dt.isoformat() if start_dt else "",
                "end_datetime":         end_dt.isoformat()   if end_dt   else "",
                "venue":                venue,
                "lat":                  vi["lat"],
                "lon":                  vi["lon"],
                "geohash":              encode_geohash(vi["lat"], vi["lon"]),
                "estimated_attendance": vi["cap"],
                "is_active":            _is_active_now(start_dt, end_dt),
                "url":                  url,
            })
        except Exception:
            continue
    return events


# ─────────────────────────────────────────────────────────────────────────────
# TICKETMASTER — anlık
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ticketmaster_anlik(api_key: str, days_ahead: int = 7) -> list[dict]:
    print("[Ticketmaster] Yaklaşan etkinlikler çekiliyor...")

    today    = datetime.now()
    end_date = today + timedelta(days=days_ahead)

    params = {
        "apikey":        api_key,
        "city":          "Istanbul",
        "countryCode":   "TR",
        "startDateTime": today.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endDateTime":   end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "size":          100,
        "sort":          "date,asc",
    }
    events = []
    try:
        r = requests.get(
            "https://app.ticketmaster.com/discovery/v2/events.json",
            params=params, timeout=15
        )
        r.raise_for_status()
        items = r.json().get("_embedded", {}).get("events", [])

        for ev in items:
            vi_tm      = ev.get("_embedded", {}).get("venues", [{}])[0]
            venue_name = vi_tm.get("name", "")
            address    = vi_tm.get("address", {}).get("line1", "")

            if not is_besiktas(venue_name, address):
                continue

            start_info = ev.get("dates", {}).get("start", {})
            date_str   = start_info.get("localDate", "")
            time_str   = (start_info.get("localTime") or "")[:5]
            category   = _guess_category(ev.get("name", ""))
            vi         = venue_info(venue_name)

            start_dt = None
            end_dt   = None
            if date_str and time_str:
                try:
                    start_dt = datetime.strptime(
                        f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
                    )
                    end_dt = _estimate_end(start_dt, category)
                except Exception:
                    pass

            loc = vi_tm.get("location", {})
            lat = float(loc.get("latitude",  vi["lat"]) or vi["lat"])
            lon = float(loc.get("longitude", vi["lon"]) or vi["lon"])

            events.append({
                "source":               "Ticketmaster",
                "name":                 ev.get("name", ""),
                "category":             category,
                "start_date_hour":      _date_hour(start_dt, date_str),
                "end_date_hour":        _date_hour(end_dt, ""),
                "start_datetime":       start_dt.isoformat() if start_dt else "",
                "end_datetime":         end_dt.isoformat()   if end_dt   else "",
                "venue":                venue_name,
                "lat":                  lat,
                "lon":                  lon,
                "geohash":              encode_geohash(lat, lon),
                "estimated_attendance": vi["cap"],
                "is_active":            _is_active_now(start_dt, end_dt),
                "url":                  ev.get("url", ""),
            })

    except Exception as e:
        print(f"  [HATA] {e}")

    print(f"  → {len(events)} etkinlik")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ENTEGRASYON FONKSİYONLARI
# ─────────────────────────────────────────────────────────────────────────────

def _is_active_now(start_dt, end_dt) -> int:
    """Etkinlik şu an devam ediyor mu? 1/0"""
    now = datetime.now()
    if start_dt and end_dt:
        return int(start_dt <= now <= end_dt)
    return 0


def get_active_events(tm_key: str = "", days_ahead: int = 7) -> list[dict]:
    """
    Model inference için ana fonksiyon.
    Tüm aktif + yaklaşan Beşiktaş etkinliklerini döner.

    Kullanım:
        from anlik_etkinlik_scraper import get_active_events
        events = get_active_events(tm_key="...")
    """
    events = scrape_radar_anlik(days_ahead)
    if tm_key:
        events += fetch_ticketmaster_anlik(tm_key, days_ahead)

    # Yineleme temizle — tarih kısmı start_date_hour'un ilk 10 karakteri
    seen, out = set(), []
    for e in events:
        key = (e.get("start_date_hour", "")[:10], e["venue"][:20].lower())
        if key not in seen:
            seen.add(key)
            out.append(e)

    out.sort(key=lambda e: e.get("start_datetime") or e.get("start_date_hour") or "")
    return out


def is_event_now(lat: float = 41.04, lon: float = 29.01,
                 radius_km: float = 2.0,
                 tm_key: str = "") -> int:
    """
    LSTM feature olarak kullanım için.
    Verilen koordinat yakınında şu an aktif etkinlik var mı?
    1 = var, 0 = yok

    Kullanım:
        from anlik_etkinlik_scraper import is_event_now
        feature = is_event_now(lat=41.039, lon=29.010)  # Vodafone Park
    """
    events = get_active_events(tm_key)
    now    = datetime.now()

    for e in events:
        if e.get("is_active") != 1:
            # Bitiş saati geçmemiş ve başlamış mı?
            try:
                s = datetime.fromisoformat(e["start_datetime"])
                en = datetime.fromisoformat(e["end_datetime"])
                if not (s <= now <= en):
                    continue
            except Exception:
                continue

        # Mesafe kontrolü (yaklaşık, Haversine yerine bbox)
        dlat = abs(e["lat"] - lat) * 111
        dlon = abs(e["lon"] - lon) * 85
        dist = (dlat**2 + dlon**2) ** 0.5
        if dist <= radius_km:
            return 1

    return 0


def get_event_features(tm_key: str = "") -> dict:
    """
    LSTM için hazır feature sözlüğü döner.
    Her 10 dakikada bir çağrılabilir.

    Dönen dict örneği:
    {
      "is_event_now":          1,
      "event_count_today":     3,
      "max_attendance_today":  42684,
      "hours_to_next_event":   1.5,
      "active_event_name":     "Beşiktaş Maçı",
    }
    """
    events = get_active_events(tm_key, days_ahead=2)
    now    = datetime.now()
    today  = now.strftime("%Y-%m-%d")

    today_events   = [e for e in events if e.get("start_date_hour", "")[:10] == today]
    active_events  = [e for e in today_events if e.get("is_active") == 1]
    future_events  = []

    for e in events:
        try:
            s = datetime.fromisoformat(e["start_datetime"])
            if s > now:
                future_events.append((s, e))
        except Exception:
            pass

    future_events.sort(key=lambda x: x[0])
    next_event_hours = (
        (future_events[0][0] - now).total_seconds() / 3600
        if future_events else 99.0
    )

    return {
        "is_event_now":         int(len(active_events) > 0),
        "event_count_today":    len(today_events),
        "max_attendance_today": max(
            (e.get("estimated_attendance", 0) for e in today_events), default=0
        ),
        "hours_to_next_event":  round(next_event_hours, 2),
        "active_event_name":    active_events[0]["name"] if active_events else "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Anlık Beşiktaş Etkinlik Scraper — Model entegrasyonu"
    )
    parser.add_argument("--days",   type=int, default=7,
                        help="Kaç gün ileriye bak (varsayılan: 7)")
    parser.add_argument("--tm-key", default="",
                        help="Ticketmaster API key")
    parser.add_argument("--json",   action="store_true",
                        help="JSON formatında çıktı ver")
    parser.add_argument("--check",  action="store_true",
                        help="Şu an etkinlik var mı? (0/1)")
    parser.add_argument("--features", action="store_true",
                        help="LSTM feature sözlüğünü göster")
    parser.add_argument("--out",    default="",
                        help="CSV çıktı dosyası (opsiyonel)")
    args = parser.parse_args()

    # Sadece aktif/mevcut etkinlik kontrolü
    if args.check:
        flag = is_event_now(tm_key=args.tm_key)
        print(f"Şu an etkinlik var mı: {flag}  ({'EVET' if flag else 'HAYIR'})")
        return

    # LSTM feature sözlüğü
    if args.features:
        features = get_event_features(tm_key=args.tm_key)
        print(json.dumps(features, ensure_ascii=False, indent=2))
        return

    # Tam liste
    events = get_active_events(tm_key=args.tm_key, days_ahead=args.days)

    if args.json:
        print(json.dumps(events, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  YAKLAŞAN BEŞİKTAŞ ETKİNLİKLERİ (önümüzdeki {args.days} gün)")
        print(f"{'='*60}")
        for i, e in enumerate(events, 1):
            aktif = " ◀ ŞU AN AKTİF" if e.get("is_active") else ""
            print(f"\n{i:>2}. {e['name']}{aktif}")
            print(f"    {e['start_date_hour']} – {e['end_date_hour']}")
            print(f"    {e['venue']}  [{e['geohash']}]")
            print(f"    Tahmini katılım: {e['estimated_attendance']:,}")
        print(f"\n{'='*60}")
        print(f"  Toplam: {len(events)} etkinlik")
        print(f"{'='*60}")

    if args.out:
        pd.DataFrame(events).to_parquet(args.out, index=False, engine="pyarrow")
        print(f"\n[CSV] {args.out}")


if __name__ == "__main__":
    main()