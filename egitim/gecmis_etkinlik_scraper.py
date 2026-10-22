"""
Geçmiş Etkinlik Scraper — LSTM Model Eğitimi İçin
===================================================
Beşiktaş ilçesindeki geçmiş etkinlikleri birden fazla kaynaktan çeker
ve LSTM modeline beslenmeye hazır tek bir CSV üretir.

Kaynaklar (yalnızca gerçekten GEÇMİŞ veri sağlayan kaynaklar kullanılıyor):
  1. Setlist.fm API    — Geçmiş konserler, kullanıcı katkılı arşiv (--sf-key ile, ÜCRETSİZ)
  2. Mackolik           — arsiv.mackolik.com üzerinden geçmiş BJK ev sahibi maçları (otomatik)
  3. Ticketmaster API  — opsiyonel (--tm-key), ancak bu bir "keşif/satış" API'si
                          olduğundan geçmiş etkinlikler için genelde boş sonuç döner.

  NOT: Radar Türkiye desteği kod tabanından kaldırıldı. Radar sadece
  güncel/yaklaşan etkinlikleri listeliyor — geçmişe dönük bir arşivi yok,
  bu yüzden bu script'in amacına (geçmiş veri) hizmet etmiyordu.

Kurulum:
    pip install requests beautifulsoup4 pandas pyarrow

Kullanım:
    python gecmis_etkinlik_scraper.py --sf-key KEY              # Setlist.fm + Mackolik
    python gecmis_etkinlik_scraper.py                            # sadece Mackolik
    python gecmis_etkinlik_scraper.py --demo                    # internet gerekmez
    python gecmis_etkinlik_scraper.py --start 2022-01-01        # tarih aralığı
                                      --end   2024-12-31         --sf-key KEY

Çıktı sütunları (LSTM için):
    event_date, start_datetime, end_datetime,
    start_time, end_time,
    name, category, venue, lat, lon,
    estimated_attendance, source
"""

import re
import time
import json
import argparse
import random
from datetime import datetime as _dt
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# MEKAN VERİTABANI — koordinat + tahmini kapasite
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# GEOHASH — harici kütüphane gerektirmeyen saf Python implementasyonu
# anlik_etkinlik_scraper.py ile AYNI algoritma/hassasiyet — join tutarlılığı
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


def _date_hour(dt, fallback_date: str = "") -> str:
    """
    Tarih + saati tek sütunda birleştirir, dakika bilgisi YOK.
    dt varsa  -> "YYYY-MM-DD HH"
    dt yoksa  -> sadece fallback_date
    """
    if dt:
        return dt.strftime("%Y-%m-%d %H")
    return fallback_date


VENUE_DB = {
    "vodafone park":                  {"lat": 41.0390, "lon": 29.0100, "cap": 42684},
    "tüpraş stadyumu":                {"lat": 41.0390, "lon": 29.0100, "cap": 42684},
    "beşiktaş tüpraş stadyumu":       {"lat": 41.0390, "lon": 29.0100, "cap": 42684},
    "zorlu psm":                      {"lat": 41.0681, "lon": 29.0116, "cap": 2350},
    "zorlu performans sanatları":     {"lat": 41.0681, "lon": 29.0116, "cap": 2350},
    "harbiye cemil topuzlu":          {"lat": 41.0487, "lon": 28.9887, "cap": 5000},
    "ibb harbiye":                    {"lat": 41.0487, "lon": 28.9887, "cap": 5000},
    "if performance hall beşiktaş":   {"lat": 41.0435, "lon": 29.0068, "cap": 800},
    "if performance hall":            {"lat": 41.0435, "lon": 29.0068, "cap": 800},
    "sahne beşiktaş":                 {"lat": 41.0428, "lon": 29.0050, "cap": 500},
    "çarşı pub":                      {"lat": 41.0410, "lon": 29.0045, "cap": 300},
    "mentalist pub beşiktaş":         {"lat": 41.0415, "lon": 29.0048, "cap": 250},
    "çırağan palace":                 {"lat": 41.0467, "lon": 29.0097, "cap": 700},
    "swissotel":                      {"lat": 41.0487, "lon": 29.0009, "cap": 800},
    "beşiktaş kültür merkezi":        {"lat": 41.0428, "lon": 29.0050, "cap": 800},
    "bkm sahne":                      {"lat": 41.0428, "lon": 29.0050, "cap": 580},
    "default_besiktas":               {"lat": 41.0402, "lon": 29.0097, "cap": 500},
}

BESIKTAS_KW = [
    "beşiktaş", "besiktas", "vodafone", "tüpraş", "tupras",
    "zorlu", "çırağan", "ciragan", "harbiye", "ortaköy",
    "balmumcu", "bebek", "akaretler", "if performance",
    "sahne beşiktaş", "çarşı pub", "mentalist",
]

def venue_info(name: str) -> dict:
    n = name.lower().strip()
    for key, data in VENUE_DB.items():
        if key in n or n in key:
            return data
    return VENUE_DB["default_besiktas"]

def is_besiktas(venue: str, city: str = "") -> bool:
    combined = (venue + " " + city).lower()
    return any(kw in combined for kw in BESIKTAS_KW)

def estimate_end(start_dt: datetime, category: str, venue: str) -> datetime:
    """Kategoriye göre tahmini bitiş saati."""
    offsets = {
        "mac": 2.5, "spor": 2.5,
        "konser": 3.0, "festival": 4.0,
        "tiyatro": 2.5, "stand-up": 2.0,
        "sergi": 3.0, "diger": 2.5,
    }
    cat = category.lower()
    for key, hours in offsets.items():
        if key in cat:
            return start_dt + timedelta(hours=hours)
    return start_dt + timedelta(hours=2.5)


# ─────────────────────────────────────────────────────────────────────────────
# 1. RADAR TÜRKİYE — Playwright ile JS render
# ─────────────────────────────────────────────────────────────────────────────

RADAR_SEARCH_URL = "https://www.radarturkiye.com/ara?q=be%C5%9Fikta%C5%9F&city=istanbul"
RADAR_BASE       = "https://www.radarturkiye.com"

def scrape_radar_turkiye(
    start_date: str = "2020-01-01",
    end_date:   str = "2024-12-31",
) -> list[dict]:
    """
    Radar Türkiye'den Beşiktaş etkinliklerini çeker.
    Playwright kullanır (JS render gerekli).
    """
    print("\n[Radar Türkiye] Scrape başlıyor...")
    try:
        from playwright.sync_api import sync_playwright
        return _radar_playwright(start_date, end_date)
    except ImportError:
        print("  [UYARI] Playwright kurulu değil, requests deneniyor...")
        return _radar_requests(start_date, end_date)
    except Exception as e:
        print(f"  [HATA] {e}")
        return []


def _radar_playwright(start_date: str, end_date: str) -> list[dict]:
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup

    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          "AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36")

    # Beşiktaş mekânlarını tek tek ara
    venues_to_search = [
        "beşiktaş", "vodafone park", "zorlu psm",
        "if performance hall beşiktaş", "harbiye",
    ]

    events = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_context(user_agent=UA).new_page()

        for venue_kw in venues_to_search:
            url = f"{RADAR_BASE}/ara?q={requests.utils.quote(venue_kw)}"
            print(f"  → '{venue_kw}' aranıyor...")
            try:
                page.goto(url, timeout=20000)
                page.wait_for_load_state("networkidle", timeout=15000)
                # Sayfayı scroll et — lazy load için
                for _ in range(5):
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    time.sleep(0.6)

                soup  = BeautifulSoup(page.content(), "html.parser")
                found = _parse_radar_cards(soup)
                # Tarih filtresi
                found = _filter_by_date(found, start_date, end_date)
                events.extend(found)
                print(f"     {len(found)} etkinlik")
            except Exception as e:
                print(f"     [HATA] {e}")
            time.sleep(1)

        browser.close()

    return events


def _radar_requests(start_date: str, end_date: str) -> list[dict]:
    """Playwright yoksa requests fallback."""
    from bs4 import BeautifulSoup

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 Chrome/123.0.0.0 Safari/537.36",
        "Accept-Language": "tr-TR,tr;q=0.9",
    })

    venues_to_search = ["beşiktaş", "vodafone park", "zorlu psm", "harbiye"]
    events = []

    for kw in venues_to_search:
        url = f"{RADAR_BASE}/ara?q={requests.utils.quote(kw)}"
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            soup  = BeautifulSoup(r.text, "html.parser")
            found = _parse_radar_cards(soup)
            found = _filter_by_date(found, start_date, end_date)
            events.extend(found)
            print(f"  → '{kw}': {len(found)} etkinlik")
            time.sleep(1.2)
        except Exception as e:
            print(f"  → '{kw}': [HATA] {e}")

    return events


def _parse_radar_cards(soup) -> list[dict]:
    """Radar Türkiye kart HTML'ini parse et."""
    events = []

    # Kart seçiciler — site güncellenirse burası değişebilir
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
            for tag in ["h1", "h2", "h3", "[class*='title']", "[class*='name']"]:
                el = card.select_one(tag)
                if el:
                    name = el.get_text(strip=True)
                    break
            if not name or len(name) < 3:
                continue

            # Tarih ve saat — "13 Nisan 2026, Pazartesi · 20:30" formatı
            date_str, time_str = "", ""
            for tag in ["time", "[class*='date']", "[class*='Date']",
                        "[class*='tarih']", "p", "span"]:
                els = card.select(tag)
                for el in els:
                    text = el.get_text(strip=True)
                    # Saat
                    m = re.search(r'\b(\d{1,2}):(\d{2})\b', text)
                    if m and not time_str:
                        time_str = f"{int(m.group(1)):02d}:{m.group(2)}"
                    # Tarih
                    if not date_str:
                        d = _parse_tr_date(text)
                        if d:
                            date_str = d

            # Mekan
            venue = ""
            for tag in ["[class*='venue']", "[class*='mekan']",
                        "[class*='location']", "[class*='place']"]:
                el = card.select_one(tag)
                if el:
                    venue = el.get_text(strip=True)
                    break

            # Link
            link = card if card.name == "a" else card.find("a")
            url  = ""
            if link and link.get("href"):
                href = link["href"]
                url  = href if href.startswith("http") else f"{RADAR_BASE}{href}"

            # Kategori (URL veya etiket'ten)
            category = _guess_category(name, url)

            if not is_besiktas(venue, name):
                continue

            vi = venue_info(venue)
            start_dt = _to_datetime(date_str, time_str)
            end_dt   = estimate_end(start_dt, category, venue) if start_dt else None

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
                "url":                  url,
            })
        except Exception:
            continue

    return events


# ─────────────────────────────────────────────────────────────────────────────
# 2. TICKETMASTER — geçmiş etkinlikler
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ticketmaster_history(
    api_key:    str,
    start_date: str = "2020-01-01",
    end_date:   str = "2024-12-31",
) -> list[dict]:
    print("\n[Ticketmaster] Geçmiş etkinlikler çekiliyor...")

    base   = "https://app.ticketmaster.com/discovery/v2/events.json"
    events = []
    page   = 0

    while True:
        params = {
            "apikey":          api_key,
            "city":            "Istanbul",
            "countryCode":     "TR",
            "startDateTime":   f"{start_date}T00:00:00Z",
            "endDateTime":     f"{end_date}T23:59:59Z",
            "size":            200,
            "page":            page,
            "sort":            "date,asc",
        }
        try:
            r = requests.get(base, params=params, timeout=15)
            r.raise_for_status()
            data  = r.json()
            items = data.get("_embedded", {}).get("events", [])
            if not items:
                break

            for ev in items:
                venue_info_tm = ev.get("_embedded", {}).get("venues", [{}])[0]
                venue_name    = venue_info_tm.get("name", "")
                address       = venue_info_tm.get("address", {}).get("line1", "")

                if not is_besiktas(venue_name, address):
                    continue

                start_info = ev.get("dates", {}).get("start", {})
                date_str   = start_info.get("localDate", "")
                time_str   = (start_info.get("localTime") or "")[:5]
                clf        = ev.get("classifications", [{}])
                category   = clf[0].get("genre", {}).get("name") or \
                             clf[0].get("segment", {}).get("name") or "Etkinlik"

                vi       = venue_info(venue_name)
                start_dt = _to_datetime(date_str, time_str)
                end_dt   = estimate_end(start_dt, category, venue_name) if start_dt else None

                loc = venue_info_tm.get("location", {})
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
                    "url":                  ev.get("url", ""),
                })

            total_pages = data.get("page", {}).get("totalPages", 1)
            page += 1
            if page >= total_pages:
                break
            time.sleep(0.3)

        except Exception as e:
            print(f"  [HATA] sayfa {page}: {e}")
            break

    print(f"  → {len(events)} Beşiktaş etkinliği")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# 3. SETLIST.FM — geçmiş konserler (2015'e kadar)
# Ücretsiz API key: setlist.fm/settings/api
# ─────────────────────────────────────────────────────────────────────────────

SETLIST_VENUES = [
    "Vodafone Park", "Zorlu Center PSM",
    "IF Performance Hall", "Harbiye Cemil Topuzlu",
]

def fetch_setlistfm(
    api_key:    str = 	"dWqSQA5hA4HJtnaqV8krv_R7sB6k6VUuiRxU",
    start_date: str = "2020-01-01",
    end_date:   str = "2099-12-31",
) -> list[dict]:
    """
    setlist.fm REST API — gerçek, kullanıcı katkılı geçmiş konser arşivi.
    Ücretsiz API key gerekli (setlist.fm hesabı > API Settings). Ticari
    olmayan kullanım için ücretsiz; hız limiti standart anahtarlarda
    ~2 istek/sn, günde ~1440 istek.
    Venue setlist'leri varsayılan olarak EN YENİDEN EN ESKİYE sıralanır;
    bu yüzden start_date'in altına inince döngüden çıkıyoruz, ama
    end_date'in üstündeki (çok yakın tarihli) kayıtları da atlıyoruz.
    """
    print("\n[Setlist.fm] Geçmiş konserler çekiliyor...")

    headers = {
        "x-api-key":  api_key,
        "Accept":     "application/json",
    }
    events = []

    for venue_name in SETLIST_VENUES:
        # Önce mekan ID'sini bul
        r = requests.get(
            "https://api.setlist.fm/rest/1.0/search/venues",
            params={"name": venue_name, "cityName": "Istanbul"},
            headers=headers, timeout=10,
        )
        if r.status_code == 401:
            print("  [HATA] API key geçersiz (401 Unauthorized). "
                  "--sf-key'e gerçek setlist.fm anahtarınızı verdiğinizden emin olun.")
            break  # anahtar geçersizse diğer mekanlar için de deneme boşuna
        if r.status_code != 200:
            print(f"  → '{venue_name}': mekan araması başarısız (HTTP {r.status_code})")
            continue
        venues = r.json().get("venue", [])
        if not venues:
            print(f"  → '{venue_name}': setlist.fm'de bu isimde mekan bulunamadı")
            continue

        venue_id = venues[0]["id"]
        found_name = venues[0].get("name", venue_name)
        page = 1
        venue_count = 0

        while True:
            r2 = requests.get(
                f"https://api.setlist.fm/rest/1.0/venue/{venue_id}/setlists",
                params={"p": page},
                headers=headers, timeout=10,
            )
            if r2.status_code != 200:
                break

            data     = r2.json()
            setlists = data.get("setlist", [])
            if not setlists:
                break

            for sl in setlists:
                date_str = sl.get("eventDate", "")  # "DD-MM-YYYY"
                try:
                    dt = datetime.strptime(date_str, "%d-%m-%Y")
                    iso_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    continue

                if iso_date < start_date:
                    break
                if iso_date > end_date:
                    continue  # bu sayfada daha eski kayıtlar da olabilir, devam et

                artist   = sl.get("artist", {}).get("name", "")
                vi       = venue_info(venue_name)
                start_dt = datetime.combine(dt.date(), datetime.min.time().replace(hour=20))
                end_dt   = start_dt + timedelta(hours=3)

                events.append({
                    "source":               "Setlist.fm",
                    "name":                 f"{artist} Konseri",
                    "category":             "Konser",
                    "start_date_hour":      _date_hour(start_dt, iso_date),
                    "end_date_hour":        _date_hour(end_dt, ""),
                    "start_datetime":       start_dt.isoformat(),
                    "end_datetime":         end_dt.isoformat(),
                    "venue":                venue_name,
                    "lat":                  vi["lat"],
                    "lon":                  vi["lon"],
                    "geohash":              encode_geohash(vi["lat"], vi["lon"]),
                    "estimated_attendance": vi["cap"],
                    "url":                  sl.get("url", ""),
                })
                venue_count += 1

            total = int(data.get("total", 0))
            items = int(data.get("itemsPerPage", 20))
            if page * items >= total:
                break
            page += 1
            time.sleep(0.6)

        print(f"  → '{venue_name}' (setlist.fm'de: '{found_name}'): {venue_count} konser")

    print(f"  → toplam {len(events)} konser")
    return deduplicate(events)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MACKOLİK — BJK geçmiş maç fikstürü
# ─────────────────────────────────────────────────────────────────────────────

MACKOLIK_BASE     = "https://arsiv.mackolik.com"
MACKOLIK_TEAM_ID   = 3            # Beşiktaş — arsiv.mackolik.com/Takim/3/Besiktas
MACKOLIK_TEAM_SLUG = "Besiktas"   # maç linklerindeki takım kısaltması (örn: /Mac/123/Besiktas-Eyupspor)

# Bu başlıkların altındaki maçlar dahil EDİLMEZ — hazırlık/dostluk maçları
# çoğunlukla yurt dışı kamplarda / nötr sahalarda oynanıyor, Vodafone Park'ta değil.
MACKOLIK_EXCLUDE_SECTIONS = ["hazırlık", "kulüpler", "hazirlik"]


def _mackolik_is_home_slug(slug: str) -> bool | None:
    """
    Maç linkindeki slug'a bakarak Beşiktaş'ın ev sahibi olup olmadığını belirler.
    Slug formatı: {EvSahibi}-{Deplasman} — örn. 'Besiktas-Eyupspor' (ev),
    'Alanyaspor-Besiktas' (deplasman). Beşiktaş'ın kendi slug'ı tek parça
    ('Besiktas') olduğu için baş/son eşleşmesi güvenilir.
    """
    if slug.startswith(MACKOLIK_TEAM_SLUG + "-"):
        return True
    if slug.endswith("-" + MACKOLIK_TEAM_SLUG):
        return False
    return None  # emin değiliz — bu maçı atlayacağız


def scrape_bjk_fixtures(
    start_year: int = 2020,
    end_year:   int = 2024,
) -> list[dict]:
    """
    arsiv.mackolik.com'dan Beşiktaş'ın geçmiş EV SAHİBİ maçlarını (Vodafone Park /
    Tüpraş Stadyumu) çeker. Sayfa yapısı doğrulandı:
      https://arsiv.mackolik.com/Takim/3/Besiktas/{yıl}/{yıl+1}
    Maç linkleri şu formatta: /Mac/{id}/{EvSahibiSlug}-{DeplasmanSlug}
    NOT: Bu fonksiyon canlı siteye bağımlıdır. Mackolik zaman zaman HTML
    yapısını değiştirebilir — çalışmazsa önce tek bir sezonu elle
    (requests + view-source) kontrol edip select0r'ları güncelleyin.
    """
    from bs4 import BeautifulSoup

    print("\n[Mackolik] BJK geçmiş EV SAHİBİ maçları çekiliyor (arsiv.mackolik.com)...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36"),
        "Accept-Language": "tr-TR,tr;q=0.9",
    })

    events = []
    vi = venue_info("vodafone park")
    date_re = re.compile(r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b')
    mac_href_re = re.compile(r'/Mac/(\d+)/([A-Za-z0-9İıŞşĞğÜüÖöÇç-]+)')

    for season in range(start_year, end_year + 1):
        url = f"{MACKOLIK_BASE}/Takim/{MACKOLIK_TEAM_ID}/{MACKOLIK_TEAM_SLUG}/{season}/{season+1}"
        season_count = 0
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # Maç listesinin olduğu ana konteyner — "Maçlar" sekmesi
            container = soup.select_one("#matchesTab") or soup

            current_section = ""
            # container.descendants: her düğümü TAM OLARAK BİR KEZ, belge sırasına
            # göre dolaşır. find_all(True) + iç find() kombinasyonu aynı linki
            # birden fazla ata seviyesinde tekrar tekrar işleyip kopya kayıt
            # üretiyordu — bu yüzden düz, tek geçişli descendants kullanıyoruz.
            for el in container.descendants:
                if not getattr(el, "name", None):
                    continue  # NavigableString vs. — sadece Tag'lerle ilgileniyoruz

                # Turnuva/bölüm başlığı mı? (Puan-Durumu linkine sahip başlıklar)
                if el.name == "a" and el.get("href") and "/Puan-Durumu/" in el["href"]:
                    txt = el.get_text(strip=True)
                    if txt:
                        current_section = txt
                    continue

                # Maç linki mi?
                if el.name != "a" or not el.get("href"):
                    continue
                href = el["href"]
                m = mac_href_re.search(href)
                if not m:
                    continue

                # Hazırlık/dostluk maçlarını atla — Vodafone Park dışı olma ihtimali yüksek
                if any(kw in current_section.lower() for kw in MACKOLIK_EXCLUDE_SECTIONS):
                    continue

                is_home = _mackolik_is_home_slug(m.group(2))
                if not is_home:
                    continue  # deplasman maçı ya da belirsiz — atla

                # Tarihi bulmak için en yakın "satır" atasını arıyoruz — ata
                # zincirinde tarih deseni (DD.MM.YYYY) içeren ilk elemanı kullan.
                row_text = ""
                node = el
                for _ in range(4):  # en fazla 4 seviye yukarı çık
                    node = node.parent
                    if node is None:
                        break
                    row_text = node.get_text(" ", strip=True)
                    if date_re.search(row_text):
                        break
                date_m = date_re.search(row_text)
                if not date_m:
                    continue

                day, mon, year = (int(x) for x in date_m.groups())
                try:
                    dt = datetime(year, mon, day)
                except ValueError:
                    continue
                iso = dt.strftime("%Y-%m-%d")

                # Saat verisi bu sayfada genelde yok — sezon boyu tipik akşam
                # maç saati olan 19:00'ı varsayılan olarak kullanıyoruz.
                time_m = re.search(r'\b(\d{1,2}):(\d{2})\b', row_text)
                time_str = f"{int(time_m.group(1)):02d}:{time_m.group(2)}" if time_m else "19:00"

                start_dt = _to_datetime(iso, time_str) or datetime(year, mon, day, 19, 0)
                end_dt   = start_dt + timedelta(hours=2.5)

                # Rakip ismi — slug'ın "Besiktas-" kısmından sonrasını okunabilir hale getir
                opponent = m.group(2)[len(MACKOLIK_TEAM_SLUG) + 1:].replace("-", " ")
                name = f"Beşiktaş - {opponent} Maçı".strip() if opponent else "Beşiktaş Maçı"

                match_id = m.group(1)
                events.append({
                    "source":               "Mackolik",
                    "name":                 name,
                    "category":             "Spor",
                    "start_date_hour":      _date_hour(start_dt, iso),
                    "end_date_hour":        _date_hour(end_dt, ""),
                    "start_datetime":       start_dt.isoformat(),
                    "end_datetime":         end_dt.isoformat(),
                    "venue":                "Vodafone Park",
                    "lat":                  vi["lat"],
                    "lon":                  vi["lon"],
                    "geohash":              encode_geohash(vi["lat"], vi["lon"]),
                    "estimated_attendance": vi["cap"],
                    "url":                  f"{MACKOLIK_BASE}/Mac/{match_id}/{m.group(2)}",
                })
                season_count += 1

            print(f"  → {season}-{season+1} sezonu: {season_count} ev sahibi maç")
            time.sleep(1)

        except Exception as e:
            print(f"  → {season}-{season+1}: [HATA] {e}")

    return deduplicate(events)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO VERİSİ
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo(start_date: str = "2020-01-01", end_date: str = "2024-12-31") -> list[dict]:
    """Gerçekçi geçmiş etkinlik verisi üretir (internet gerekmez)."""
    random.seed(42)
    events = []

    venues = [
        ("Vodafone Park",            "Spor / Maç",  42684, 41.0390, 29.0100, "19:00", 2.5),
        ("Vodafone Park",            "Konser",       42684, 41.0390, 29.0100, "20:00", 3.0),
        ("Zorlu PSM",                "Konser",        2350, 41.0681, 29.0116, "20:00", 3.0),
        ("Harbiye Cemil Topuzlu",    "Konser",        5000, 41.0487, 28.9887, "21:00", 3.0),
        ("IF Performance Hall",      "Konser",         800, 41.0435, 29.0068, "21:00", 2.5),
        ("Sahne Beşiktaş",           "Stand-up",       500, 41.0428, 29.0050, "20:30", 2.0),
    ]

    artists = [
        "Sertab Erener", "Mor ve Ötesi", "Duman", "Teoman",
        "Ajda Pekkan", "Ceza", "Şehinşah", "Sansar Salvo",
        "Haluk Levent", "Sıla", "Gripin", "Hayko Cepkin",
    ]

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
    current  = start_dt

    while current <= end_dt:
        # Haftada ~2-3 etkinlik
        if random.random() < 0.35:
            venue, cat, cap, lat, lon, t, dur = random.choice(venues)
            artist = random.choice(artists) if "Konser" in cat else ""
            name   = f"{artist} Konseri" if artist else (
                "Beşiktaş Maçı" if "Maç" in cat else f"Beşiktaş {cat}"
            )
            h, m   = int(t.split(":")[0]), int(t.split(":")[1])
            s_dt   = current.replace(hour=h, minute=m, second=0)
            e_dt   = s_dt + timedelta(hours=dur)

            events.append({
                "source":               "Demo",
                "name":                 name,
                "category":             cat,
                "start_date_hour":      _date_hour(s_dt, current.strftime("%Y-%m-%d")),
                "end_date_hour":        _date_hour(e_dt, ""),
                "start_datetime":       s_dt.isoformat(),
                "end_datetime":         e_dt.isoformat(),
                "venue":                venue,
                "lat":                  lat,
                "lon":                  lon,
                "geohash":              encode_geohash(lat, lon),
                "estimated_attendance": int(cap * random.uniform(0.6, 1.0)),
                "url":                  "",
            })
        current += timedelta(days=1)

    return events


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────

TR_MONTHS = {
    "ocak": 1, "şubat": 2, "mart": 3, "nisan": 4,
    "mayıs": 5, "haziran": 6, "temmuz": 7, "ağustos": 8,
    "eylül": 9, "ekim": 10, "kasım": 11, "aralık": 12,
}

def _parse_tr_date(text: str) -> str:
    """'13 Nisan 2026' → '2026-04-13'"""
    m = re.search(
        r'(\d{1,2})\s+([A-Za-zğışüöçĞİŞÜÖÇ]+)\s+(\d{4})', text
    )
    if m:
        day  = int(m.group(1))
        mon  = TR_MONTHS.get(m.group(2).lower())
        year = int(m.group(3))
        if mon:
            return f"{year}-{mon:02d}-{day:02d}"
    m2 = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
    if m2:
        return m2.group(0)
    return ""

def _to_datetime(date_str: str, time_str: str) -> datetime | None:
    try:
        t = time_str or "00:00"
        return datetime.strptime(f"{date_str} {t}", "%Y-%m-%d %H:%M")
    except Exception:
        return None

def _filter_by_date(events: list, start: str, end: str) -> list:
    return [
        e for e in events
        if start <= e.get("start_date_hour", e.get("event_date", "9999"))[:10] <= end
    ]

def _guess_category(name: str, url: str = "") -> str:
    combined = (name + " " + url).lower()
    if any(k in combined for k in ["konser", "concert", "music"]):  return "Konser"
    if any(k in combined for k in ["tiyatro", "theatre", "oyun"]):  return "Tiyatro"
    if any(k in combined for k in ["stand-up", "standup", "komedi"]): return "Stand-up"
    if any(k in combined for k in ["maç", "mac", "futbol", "spor"]): return "Spor"
    if any(k in combined for k in ["festival", "fest"]):             return "Festival"
    return "Etkinlik"

def deduplicate(events: list) -> list:
    """Aynı gün + aynı mekan + benzer isim → tek kayıt."""
    seen = set()
    out  = []
    for e in events:
        # start_date_hour veya event_date — her iki formata uyumlu
        tarih = e.get("start_date_hour", e.get("event_date", ""))[:10]
        key   = (tarih, e.get("venue", "")[:20].lower(),
                 e.get("name", "")[:15].lower())
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Geçmiş etkinlik scraper — LSTM eğitim verisi"
    )
    parser.add_argument("--start",  default="2020-01-01", help="Başlangıç tarihi")
    parser.add_argument("--end",    default="2024-12-31", help="Bitiş tarihi")
    parser.add_argument("--tm-key", default="",  help="Ticketmaster API key")
    parser.add_argument("--sf-key", default="",  help="Setlist.fm API key")
    parser.add_argument("--demo",   action="store_true", help="Demo modu")
    parser.add_argument("--out",    default="besiktas_gecmis_etkinlikler.parquet")
    args = parser.parse_args()

    all_events = []

    if args.demo:
        print("[DEMO] Örnek geçmiş etkinlik verisi üretiliyor...")
        all_events = generate_demo(args.start, args.end)
    else:
        # NOT: Radar Türkiye kasıtlı olarak devre dışı bırakıldı. Radar sadece
        # güncel/yaklaşan etkinlikleri listeliyor — geçmişe dönük bir arşivi
        # yok, bu yüzden geçmiş veri toplama amacıyla kullanılamıyor.
        # (fonksiyon kod tabanında duruyor, isterseniz ileride GÜNCEL/gelecek
        # etkinlik toplama işi için ayrı bir script'te kullanılabilir.)

        # 1. Ticketmaster (key varsa) — UYARI: Discovery API bir "keşif/satış"
        #    sistemidir, geçmişte tamamlanmış etkinlikleri arşivlemez. Bu
        #    yüzden geçmiş tarih aralıkları için genellikle boş sonuç döner.
        #    Yine de --tm-key verilirse denenir (zararı yok).
        if args.tm_key:
            all_events += fetch_ticketmaster_history(
                args.tm_key, args.start, args.end
            )
        else:
            print("\n[Ticketmaster] --tm-key verilmedi, atlandı.")

        # 2. Setlist.fm (key varsa) — asıl geçmiş konser kaynağımız
        if args.sf_key:
            all_events += fetch_setlistfm(args.sf_key, args.start, args.end)
        else:
            print("[Setlist.fm] --sf-key verilmedi, atlandı.")

        # 3. Mackolik — BJK ev sahibi maçları (her zaman, arsiv.mackolik.com)
        start_year = int(args.start[:4])
        end_year   = int(args.end[:4])
        all_events += scrape_bjk_fixtures(start_year, end_year)

        # Mackolik sezon bazlı çalışıyor (örn. 2024/2025 sezonu Ağustos 2024 -
        # Mayıs 2025 arası sürer), bu yüzden istenen --start/--end aralığının
        # dışına taşan maçları burada kırpıyoruz.
        all_events = _filter_by_date(all_events, args.start, args.end)

    # Yineleme temizle + tarihe göre sırala
    all_events = deduplicate(all_events)
    all_events.sort(key=lambda e: e.get("start_datetime") or e.get("start_date_hour") or e.get("event_date") or "")

    df = pd.DataFrame(all_events)
    df.to_parquet(args.out, index=False, engine="pyarrow")

    # Kalite raporu
    kalite = {
        "kaynak":               "Setlist.fm + Mackolik (+ opsiyonel Ticketmaster)",
        "toplam_etkinlik":      int(len(df)),
        "kaynak_dagilimi":      {k: int(v) for k, v in df["source"].value_counts().items()} if not df.empty else {},
        "kategori_dagilimi":    {k: int(v) for k, v in df["category"].value_counts().items()} if not df.empty else {},
        "yineleme_silinen":     int(len(all_events) - len(df)),
        "koordinatsiz_silinen": int(df[["lat","lon"]].isna().any(axis=1).sum()) if not df.empty else 0,
        "tarih_araligi":        {
            "baslangic": args.start,
            "bitis":     args.end,
        },
    }
    with open("etkinlik_kalite.json", "w", encoding="utf-8") as f:
        json.dump(kalite, f, ensure_ascii=False, indent=2)
    print(f"  Kalite raporu: etkinlik_kalite.json")

    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  Toplam etkinlik : {len(df)}")
    if not df.empty:
        tarih_col = "start_date_hour" if "start_date_hour" in df.columns else "event_date"
        print(f"  Tarih aralığı   : {str(df[tarih_col].min())[:10]} → {str(df[tarih_col].max())[:10]}")
        print(f"  Kaynaklar       : {df['source'].value_counts().to_dict()}")
        print(f"  Kategoriler     : {df['category'].value_counts().to_dict()}")
    print(f"  Çıktı           : {args.out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
