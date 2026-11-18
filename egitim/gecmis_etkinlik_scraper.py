"""
Geçmiş Etkinlik Scraper — LSTM Model Eğitimi İçin
===================================================
Beşiktaş ilçesindeki geçmiş etkinlikleri birden fazla kaynaktan çeker
ve LSTM modeline beslenmeye hazır tek bir CSV üretir.

Kaynaklar (yalnızca gerçekten GEÇMİŞ veri sağlayan kaynaklar kullanılıyor):
  1. Setlist.fm API    — Geçmiş konserler, kullanıcı katkılı arşiv (--sf-key ile, ÜCRETSİZ)
  2. İBB Şehir Tiyatroları — 2017-2023 gösterim arşivi (CSV, --st-url ile veya
                          SEHIR_TIYATROLARI_CSV_URL sabitine yazılarak)
  3. Mackolik           — arsiv.mackolik.com üzerinden geçmiş BJK ev sahibi maçları (otomatik)

  EKLEME MODU: Çıktı dosyası zaten varsa içindeki kayıtlar KORUNUR, yeni
  çekilenler üstüne eklenir (aynı etkinlik iki kez yazılmaz). Sıfırdan
  yazmak için --yenile kullanın.

  NOT: Ticketmaster desteği kaldırıldı — Discovery API geçmiş etkinlikleri
  arşivlemiyor, hiç veri dönmüyordu.

  NOT: Radar Türkiye desteği kod tabanından kaldırıldı. Radar sadece
  güncel/yaklaşan etkinlikleri listeliyor — geçmişe dönük bir arşivi yok,
  bu yüzden bu script'in amacına (geçmiş veri) hizmet etmiyordu.

Kurulum:
    pip install requests beautifulsoup4 pandas pyarrow

Kullanım:
    python gecmis_etkinlik_scraper.py --sf-key KEY              # Setlist.fm + Mackolik
    python gecmis_etkinlik_scraper.py                            # sadece Mackolik
    python gecmis_etkinlik_scraper.py --st-url URL_VEYA_DOSYA   # + Şehir Tiyatroları
    python gecmis_etkinlik_scraper.py --yenile                  # mevcut dosyayı silip sıfırdan yaz
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
    "vodafone park":                  {"lat": 41.042041, "lon": 28.994508, "cap": 42684},   # koordinat kullanıcı tarafından doğrulandı
    "tüpraş stadyumu":                {"lat": 41.042041, "lon": 28.994508, "cap": 42684},   # koordinat kullanıcı tarafından doğrulandı
    "beşiktaş tüpraş stadyumu":       {"lat": 41.042041, "lon": 28.994508, "cap": 42684},   # koordinat kullanıcı tarafından doğrulandı
    # Zorlu PSM — koordinat kullanıcı tarafından doğrulandı (Levazım, Beşiktaş).
    # "zorlu center psm" anahtarı gerekli: Setlist.fm mekan adı "Zorlu Center PSM"
    # ve venue_info'nun alt dizi eşleşmesi "zorlu psm" ile bunu yakalamıyordu
    # (varsayılan Beşiktaş konumuna/500 kapasiteye düşüyordu).
    "zorlu center psm":               {"lat": 41.066812, "lon": 29.016252, "cap": 2350},
    "zorlu psm":                      {"lat": 41.066812, "lon": 29.016252, "cap": 2350},
    "zorlu performans sanatları":     {"lat": 41.066812, "lon": 29.016252, "cap": 2350},
    # Harbiye Cemil Topuzlu Açıkhava Tiyatrosu — koordinat İBB Şehir Tiyatroları
    # CSV'sindeki resmi konum (Setlist.fm ve tiyatro kayıtları aynı noktada olsun)
    "harbiye cemil topuzlu":          {"lat": 41.045777, "lon": 28.99044, "cap": 5000},
    # Şehir Tiyatroları — Harbiye Muhsin Ertuğrul (Şişli, Beşiktaş sınırında).
    # Koordinat CSV'den gelir; burası yalnızca CSV'de eksikse kullanılan yedek.
    "muhsin ertuğrul":                {"lat": 41.0466, "lon": 28.9884, "cap": 500},
    "ibb harbiye":                    {"lat": 41.045777, "lon": 28.99044, "cap": 5000},
    "if performance hall beşiktaş":   {"lat": 41.044349, "lon": 29.006474, "cap": 800},   # koordinat kullanıcı tarafından doğrulandı
    "if performance hall":            {"lat": 41.044349, "lon": 29.006474, "cap": 800},   # koordinat kullanıcı tarafından doğrulandı
    "sahne beşiktaş":                 {"lat": 41.0428, "lon": 29.0050, "cap": 500},
    "çarşı pub":                      {"lat": 41.0410, "lon": 29.0045, "cap": 300},
    "mentalist pub beşiktaş":         {"lat": 41.0415, "lon": 29.0048, "cap": 250},
    "çırağan palace":                 {"lat": 41.045100, "lon": 29.016480, "cap": 700},   # koordinat kullanıcı tarafından doğrulandı
    "swissotel":                      {"lat": 41.042487, "lon": 28.997628, "cap": 800},   # koordinat kullanıcı tarafından doğrulandı
    "beşiktaş kültür merkezi":        {"lat": 41.043729, "lon": 29.005887, "cap": 800},   # aynı bina; koordinat kullanıcı tarafından doğrulandı
    "bkm sahne":                      {"lat": 41.043729, "lon": 29.005887, "cap": 580},   # aynı bina; koordinat kullanıcı tarafından doğrulandı
    "default_besiktas":               {"lat": 41.0402, "lon": 29.0097, "cap": 500},
}

BESIKTAS_KW = [
    "beşiktaş", "besiktas", "vodafone", "tüpraş", "tupras",
    "zorlu", "çırağan", "ciragan", "harbiye", "ortaköy",
    "balmumcu", "bebek", "akaretler", "if performance",
    "sahne beşiktaş", "çarşı pub", "mentalist",
    "muhsin ertuğrul",
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
# 1. SETLIST.FM — geçmiş konserler (2015'e kadar)
# Ücretsiz API key: setlist.fm/settings/api
# ─────────────────────────────────────────────────────────────────────────────

SETLIST_VENUES = [
    "Vodafone Park", "Zorlu Center PSM",
    "IF Performance Hall", "Harbiye Cemil Topuzlu",
]

# Setlist.fm'de aranan ad → çıktıda kullanılacak ad. Aynı mekan Şehir
# Tiyatroları verisinde bu adla geçiyor; tek isimde birleştiriyoruz.
SETLIST_MEKAN_ADI = {
    "Harbiye Cemil Topuzlu": "Harbiye Cemil Topuzlu Açıkhava Tiyatrosu",
}

def fetch_setlistfm(
    api_key:    str,
    start_date: str = "2020-01-01",
    end_date:   str = "2099-12-31",
) -> list[dict]:
    """
    setlist.fm REST API — gerçek, kullanıcı katkılı geçmiş konser arşivi.
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
        time.sleep(5)  # rate limiting önlemi
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
                    "venue":                SETLIST_MEKAN_ADI.get(venue_name, venue_name),
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

    print(f"  → toplam {len(events)} sanatçı kaydı")
    konserler = _konserleri_birlestir(deduplicate(events))
    print(f"  → {len(konserler)} konser (aynı mekan + aynı gece tek etkinlik)")
    return konserler


def _konserleri_birlestir(events: list[dict]) -> list[dict]:
    """
    Setlist.fm her sanatçıyı ayrı kaydettiği için ana grup ve ön gruplar aynı
    gece ayrı konserler gibi görünüyor (ör. Arctic Monkeys + Inhaler). Aynı
    mekan + aynı başlangıçtaki Setlist.fm kayıtlarını tek etkinlikte birleştirir:
    sanatçı adları " + " ile birleşir, seyirci aynı kitle olduğu için
    TOPLANMAZ (en büyüğü alınır). Diğer kaynaklara dokunmaz. Zaten birleşmiş
    kayıtlarla tekrar çalıştırılabilir (idempotent).
    """
    out, gruplar = [], {}
    for e in events:
        if e.get("source") != "Setlist.fm":
            out.append(e)
            continue
        key = (e["venue"], e["start_datetime"])
        adlar = [a.strip() for a in re.sub(r"\s*Konseri$", "", e["name"]).split(" + ") if a.strip()]
        if key not in gruplar:
            g = dict(e)
            g["_adlar"] = []
            gruplar[key] = g
            out.append(g)
        g = gruplar[key]
        for a in adlar:
            if a not in g["_adlar"]:
                g["_adlar"].append(a)
        g["estimated_attendance"] = max(g["estimated_attendance"], e["estimated_attendance"])
    for g in gruplar.values():
        g["name"] = " + ".join(g.pop("_adlar")) + " Konseri"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. İBB ŞEHİR TİYATROLARI — 2017-2023 gösterim arşivi (CSV)
# Sütunlar: PLAY_DATE, PLAY_NAME, THEATER_NAME, PLAY_CATEGORY, IS_GUEST_PLAY,
#           PLAY_TYPE, LONGITUDE, LATITUDE, NUMBER_OF_AUDIENCE
# Her satır = tek bir gösterim.
# ─────────────────────────────────────────────────────────────────────────────

SEHIR_TIYATROLARI_CSV_URL = ""   

# PLAY_DATE'te saat yoksa kullanılacak varsayılan başlangıç saatleri
ST_VARSAYILAN_SAAT       = (20, 30)
ST_COCUK_VARSAYILAN_SAAT = (14, 0)    # çocuk oyunları genelde gündüz

# Trafik ve hava verisinin saat dilimi — etkinlik saatleri buna çevrilir
YEREL_SAAT_DILIMI = "Europe/Istanbul"


def _tr_lower(text: str) -> str:
    """Türkçe uyumlu küçük harf: 'HARBİYE' → 'harbiye' (str.lower() 'i̇' üretir)."""
    return str(text).replace("İ", "i").replace("I", "ı").lower()


def _to_float(val) -> float | None:
    """'41,0466' / '41.0466' / NaN → float veya None."""
    try:
        f = float(str(val).replace(",", "."))
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _read_csv_esnek(src: str) -> pd.DataFrame:
    """İBB CSV'leri için ayraç (; veya ,) ve kodlama (utf-8 / cp1254) otomatik."""
    son_hata = None
    for enc in ("utf-8-sig", "cp1254", "latin-1"):
        try:
            return pd.read_csv(src, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError as e:
            son_hata = e
    raise son_hata


def fetch_sehir_tiyatrolari(
    src:        str,
    start_date: str = "2020-01-01",
    end_date:   str = "2024-12-31",
) -> list[dict]:
    """
    İBB Şehir Tiyatroları gösterim CSV'sinden Beşiktaş ve çevresindeki
    (BESIKTAS_KW — örn. Harbiye Muhsin Ertuğrul) sahnelerin gösterimlerini çeker.
    src: URL veya yerel dosya yolu.
    """
    print("\n[Şehir Tiyatroları] Gösterim arşivi okunuyor...")
    try:
        raw = _read_csv_esnek(src)
    except Exception as e:
        print(f"  [HATA] CSV okunamadı: {e}")
        return []

    raw.columns = [str(c).strip().upper() for c in raw.columns]
    gerekli = {"PLAY_DATE", "PLAY_NAME", "THEATER_NAME"}
    eksik   = gerekli - set(raw.columns)
    if eksik:
        print(f"  [HATA] Beklenen sütunlar yok: {sorted(eksik)} — mevcut: {list(raw.columns)}")
        return []
    print(f"  {len(raw):,} gösterim, {raw['THEATER_NAME'].nunique()} sahne")

    # Sahne filtresi — Türkçe büyük harf (İ) sorununa karşı _tr_lower ile
    raw["_sahne_lc"] = raw["THEATER_NAME"].map(_tr_lower)
    raw = raw[raw["_sahne_lc"].map(lambda n: any(kw in n for kw in BESIKTAS_KW))]
    print(f"  Beşiktaş/çevre sahneleri: {raw['THEATER_NAME'].unique().tolist()}")

    events = []
    for _, row in raw.iterrows():
        dt = pd.to_datetime(row["PLAY_DATE"], errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(row["PLAY_DATE"], errors="coerce", dayfirst=True)
        if pd.isna(dt):
            continue

        # PLAY_DATE saat dilimi bilgisiyle (UTC, "+00:00"/"Z") geliyor; trafik ve
        # hava verisi Türkiye saatinde olduğu için Europe/Istanbul'a çevirip
        # saat dilimi bilgisini atıyoruz (ör. 17:30 UTC → 20:30). Saat dilimi
        # bilgisi olmayan değerlere dokunulmaz.
        saat_dilimli = dt.tzinfo is not None
        if saat_dilimli:
            dt = dt.tz_convert(YEREL_SAAT_DILIMI).tz_localize(None)
        dt = dt.to_pydatetime()

        iso = dt.strftime("%Y-%m-%d")
        if not (start_date <= iso <= end_date):
            continue

        # Saat bilgisi yoksa (00:00) oyun tipine göre varsayılan saat.
        # Saat dilimli değerlerde saat her zaman vardır, gece yarısı da olsa
        # varsayılan saate çevrilmez.
        if not saat_dilimli and dt.hour == 0 and dt.minute == 0:
            tip = _tr_lower(f"{row.get('PLAY_TYPE', '')} {row.get('PLAY_CATEGORY', '')}")
            h, m = ST_COCUK_VARSAYILAN_SAAT if "çocuk" in tip else ST_VARSAYILAN_SAAT
            dt = dt.replace(hour=h, minute=m)

        venue    = str(row["THEATER_NAME"]).strip()
        vi       = venue_info(row["_sahne_lc"])
        lat      = _to_float(row.get("LATITUDE"))  or vi["lat"]
        lon      = _to_float(row.get("LONGITUDE")) or vi["lon"]
        seyirci  = _to_float(row.get("NUMBER_OF_AUDIENCE"))
        end_dt   = estimate_end(dt, "tiyatro", venue)

        events.append({
            "source":               "Şehir Tiyatroları",
            "name":                 str(row["PLAY_NAME"]).strip(),
            "category":             "Tiyatro",
            "start_date_hour":      _date_hour(dt, iso),
            "end_date_hour":        _date_hour(end_dt, ""),
            "start_datetime":       dt.isoformat(),
            "end_datetime":         end_dt.isoformat(),
            "venue":                venue,
            "lat":                  lat,
            "lon":                  lon,
            "geohash":              encode_geohash(lat, lon),
            "estimated_attendance": int(seyirci) if seyirci else vi["cap"],
            "url":                  "",
        })

    print(f"  → {len(events)} gösterim satırı ({start_date} → {end_date})")
    seanslar = _seanslari_birlestir(events)
    print(f"  → {len(seanslar)} seans (aynı sahne + aynı saat tek etkinlik)")
    return seanslar


def _seanslari_birlestir(events: list[dict]) -> list[dict]:
    """
    Aynı sahnede aynı başlangıç saatindeki kayıtları tek etkinlikte birleştirir.
    Ham CSV'de aynı seans bazen iki satır (ör. seyirci 40 + 590 olarak bölünmüş)
    ya da aynı saatte iki farklı oyun olarak geçiyor; trafik açısından o saatte
    binaya gelen kişi sayısı önemli olduğu için seyirci sayıları TOPLANIR,
    oyun adları " + " ile birleştirilir.
    Aynı oyunun aynı gün farklı saatlerdeki gösterimleri (15:00 ve 20:30)
    ayrı etkinlik olarak kalır.
    """
    gruplar: dict[tuple, dict] = {}
    for e in events:
        key = (e["venue"], e["start_datetime"])
        if key not in gruplar:
            gruplar[key] = dict(e)
            gruplar[key]["_adlar"] = [e["name"]]
            continue
        g = gruplar[key]
        if e["name"] not in g["_adlar"]:
            g["_adlar"].append(e["name"])
        g["estimated_attendance"] += e["estimated_attendance"]
        if e["end_datetime"] > g["end_datetime"]:
            g["end_datetime"]  = e["end_datetime"]
            g["end_date_hour"] = e["end_date_hour"]

    out = []
    for g in gruplar.values():
        g["name"] = " + ".join(g.pop("_adlar"))
        out.append(g)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. MACKOLİK — BJK geçmiş maç fikstürü
# ─────────────────────────────────────────────────────────────────────────────

MACKOLIK_BASE     = "https://arsiv.mackolik.com"
MACKOLIK_TEAM_ID   = 3            # Beşiktaş — arsiv.mackolik.com/Takim/3/Besiktas
MACKOLIK_TEAM_SLUG = "Besiktas"   # maç linklerindeki takım kısaltması (örn: /Mac/123/Besiktas-Eyupspor)

# Bu başlıkların altındaki maçlar dahil EDİLMEZ — hazırlık/dostluk maçları
# çoğunlukla yurt dışı kamplarda / nötr sahalarda oynanıyor, Vodafone Park'ta değil.
MACKOLIK_EXCLUDE_SECTIONS = ["hazırlık", "kulüpler", "hazirlik"]

# Maç detay sayfasındaki stadyum adı bunlardan birini içermiyorsa maç ALINMAZ
# (Süper Kupa / kupa finali gibi nötr saha, yurt dışı kamp, UEFA kararıyla
# başka ülkede oynanan maçlar vb.). Stadyumun adı 2023'te Vodafone Park'tan
# Tüpraş Stadyumu'na değişti; UEFA maçlarında "Beşiktaş Park" geçebiliyor.
MACKOLIK_STADYUM_ADLARI = ["vodafone park", "vodafone arena", "tüpraş stadyumu", "beşiktaş park"]

# Pandemi nedeniyle seyircisiz oynanan dönem (Mart 2020 — 2020/21 sezonu sonu).
# Maç sayfasında seyirci yoksa ve maç bu aralıktaysa seyirci 0 kabul edilir.
SEYIRCISIZ_DONEM = ("2020-03-12", "2021-07-31")

MACKOLIK_DETAY_BEKLEME_SN = 1.2   # maç detay sayfaları arası bekleme (siteyi yormamak için)

# Son scrape_bjk_fixtures çalışmasında sezon sayfası alınamayan sezonlar
# (başlangıç yılı). main() bu sezonların eski kayıtlarını SİLMEZ.
MACKOLIK_HATALI_SEZONLAR: list[int] = []


def _mackolik_mac_detay(session, url: str, deneme: int = 3) -> dict | None:
    """
    Mackolik maç detay sayfasından turnuva, tarih/saat, durum, stadyum ve
    seyirci bilgisini okur. Sayfa alınamazsa None döner.
    Sayfa metni "Tarih : 18.08.2024 21:45 | Beşiktaş | ... | Son 5 Maç : |
    Seyirci : 32.966 kişi | MS | 4 - 2 | Antalyaspor | ... | Son 5 Maç : |
    Tüpraş Stadyumu | ..." biçiminde.
    """
    import html as _html
    t = None
    for i in range(deneme):
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            t = r.text
            break
        except Exception:
            time.sleep(3 * (i + 1))
    if t is None:
        return None

    title_m = re.search(r"<title>(.*?)</title>", t, re.S)
    title   = _html.unescape(title_m.group(1)).strip() if title_m else ""
    turnuva = re.sub(r"^.*?Maç Detayı - ", "", title).replace(" @ Mackolik.com", "").strip()

    txt = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    txt = _html.unescape(re.sub(r"<[^>]+>", " | ", txt))
    txt = re.sub(r"(\s*\|\s*)+", " | ", txt)
    i   = txt.find("Tarih :")
    blk = txt[i:i + 500] if i >= 0 else ""

    tarih_m = re.search(r"Tarih : (\d{1,2})\.(\d{1,2})\.(\d{4}) (\d{1,2}):(\d{2})", blk)
    tarih   = None
    if tarih_m:
        g, a, y, sa, dk = (int(x) for x in tarih_m.groups())
        try:
            tarih = datetime(y, a, g, sa, dk)
        except ValueError:
            tarih = None

    parcalar = blk.split("Son 5 Maç : |")
    stad = ""
    if len(parcalar) > 2:
        aday = parcalar[2].split("|")[0].strip()
        # Stadyum yoksa bu alanda "Takım Değeri :" / "Takım Karşılaştırma" gibi
        # başka bir alan oluyor — stadyum sayma
        if aday and not aday.startswith("Takım"):
            stad = aday

    # MS = normal süre, UZ = uzatma, Pen = penaltılar → oynandı.
    # "Hük." (hükmen) / "Ert." (ertelendi) → oynanmadı.
    oynandi = bool(re.search(r"\| (MS|UZ|Pen) \| \d+ - \d+ \|", blk))
    sey_m   = re.search(r"Seyirci : ([\d\.]+) kişi", blk)
    seyirci = int(sey_m.group(1).replace(".", "")) if sey_m else None

    return {"turnuva": turnuva, "tarih": tarih, "stad": stad,
            "oynandi": oynandi, "seyirci": seyirci}


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
    MACKOLIK_HATALI_SEZONLAR.clear()
    vi = venue_info("vodafone park")
    gorulen_maclar  = set()   # aynı maç linki sayfada birden fazla geçebilir
    elenen          = []      # (maç, tarih, neden) — sonda raporlanır
    seyirci_kaynagi = {}      # sayfa / seyircisiz dönem / tahmin sayıları
    mac_href_re = re.compile(r'/Mac/(\d+)/([A-Za-z0-9İıŞşĞğÜüÖöÇç-]+)')

    for season in range(start_year, end_year + 1):
        url = f"{MACKOLIK_BASE}/Takim/{MACKOLIK_TEAM_ID}/{MACKOLIK_TEAM_SLUG}/{season}/{season+1}"
        season_count = 0
        try:
            # Sezon sayfası — Mackolik ara sıra bağlantıyı kesiyor, birkaç kez dene
            for deneme in range(1, 4):
                try:
                    r = session.get(url, timeout=15)
                    r.raise_for_status()
                    break
                except Exception:
                    if deneme == 3:
                        raise
                    time.sleep(15 * deneme)
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

                match_id = m.group(1)
                if match_id in gorulen_maclar:
                    continue
                gorulen_maclar.add(match_id)
                mac_url = f"{MACKOLIK_BASE}/Mac/{match_id}/{m.group(2)}"

                # Rakip ismi — slug'ın "Besiktas-" kısmından sonrasını okunabilir hale getir
                opponent = m.group(2)[len(MACKOLIK_TEAM_SLUG) + 1:].replace("-", " ")
                name = f"Beşiktaş - {opponent} Maçı".strip() if opponent else "Beşiktaş Maçı"

                # Maç detay sayfası: gerçekten stadyumda oynandı mı, saat, seyirci
                time.sleep(MACKOLIK_DETAY_BEKLEME_SN)
                detay = _mackolik_mac_detay(session, mac_url)
                if detay is None:
                    elenen.append((name, "?", "detay sayfası alınamadı"))
                    continue
                tarih_txt = detay["tarih"].strftime("%Y-%m-%d") if detay["tarih"] else "?"
                if "hazırlık" in _tr_lower(detay["turnuva"]):
                    elenen.append((name, tarih_txt, f"hazırlık maçı ({detay['turnuva']})"))
                    continue
                if not detay["oynandi"]:
                    elenen.append((name, tarih_txt, "oynanmadı (hükmen/ertelendi)"))
                    continue
                if not any(k in _tr_lower(detay["stad"]) for k in MACKOLIK_STADYUM_ADLARI):
                    elenen.append((name, tarih_txt, f"stadyum: {detay['stad'] or 'bilinmiyor'}"))
                    continue
                if detay["tarih"] is None:
                    elenen.append((name, "?", "tarih okunamadı"))
                    continue

                start_dt = detay["tarih"]
                end_dt   = start_dt + timedelta(hours=2.5)
                iso      = start_dt.strftime("%Y-%m-%d")

                if detay["seyirci"] is not None:
                    seyirci, kaynak = detay["seyirci"], "sayfa"
                elif SEYIRCISIZ_DONEM[0] <= iso <= SEYIRCISIZ_DONEM[1]:
                    seyirci, kaynak = 0, "seyircisiz dönem"
                else:
                    seyirci, kaynak = None, "tahmin"   # döngü sonunda doldurulur
                seyirci_kaynagi[kaynak] = seyirci_kaynagi.get(kaynak, 0) + 1

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
                    "estimated_attendance": seyirci,
                    "url":                  mac_url,
                })
                season_count += 1

            print(f"  → {season}-{season+1} sezonu: {season_count} ev sahibi maç")
            time.sleep(1)

        except Exception as e:
            MACKOLIK_HATALI_SEZONLAR.append(season)
            print(f"  → {season}-{season+1}: [HATA] {e} — bu sezonun eski kayıtları korunacak")

    # Sayfada seyirci olmayan (seyircisiz dönem dışı) maçlar: bu çalışmada
    # bulunan gerçek seyirci sayılarının medyanı, hiç yoksa stadyum kapasitesi
    gercek = [e["estimated_attendance"] for e in events
              if e["estimated_attendance"] not in (None, 0)]
    tahmin = int(pd.Series(gercek).median()) if gercek else vi["cap"]
    for e in events:
        if e["estimated_attendance"] is None:
            e["estimated_attendance"] = tahmin

    print(f"  Seyirci kaynağı: {seyirci_kaynagi} (tahmin = {tahmin:,})")
    if elenen:
        print(f"  Elenen {len(elenen)} maç:")
        for ad, tarih, neden in elenen:
            print(f"    - {tarih}  {ad}: {neden}")

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
    """
    Aynı başlangıç saati + aynı mekan + benzer isim → tek kayıt.
    Eskiden yalnızca GÜNE bakılıyordu; bu, aynı oyunun aynı gün 15:00 ve
    20:30'daki iki gösteriminden ikincisini siliyordu.
    """
    seen = set()
    out  = []
    for e in events:
        # start_datetime (dakikaya kadar) → yoksa start_date_hour → yoksa event_date
        tarih = (e.get("start_datetime") or e.get("start_date_hour")
                 or e.get("event_date") or "")[:16]
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
    parser.add_argument("--sf-key", default="",  help="Setlist.fm API key")
    parser.add_argument("--st-url", default=SEHIR_TIYATROLARI_CSV_URL,
                        help="Şehir Tiyatroları CSV URL'si veya yerel dosya yolu")
    parser.add_argument("--demo",   action="store_true", help="Demo modu")
    parser.add_argument("--out",    default="besiktas_gecmis_etkinlikler.parquet")
    parser.add_argument("--yenile", action="store_true",
                        help="Mevcut çıktıyı KORUMA, sıfırdan yaz (varsayılan: üstüne ekle)")
    args = parser.parse_args()

    all_events = []
    yenilenen_kaynaklar: set[str] = set()   # eski kayıtları yenileriyle değiştirilecek kaynaklar

    if args.demo:
        print("[DEMO] Örnek geçmiş etkinlik verisi üretiliyor...")
        all_events = generate_demo(args.start, args.end)
    else:
        # NOT: Radar Türkiye kasıtlı olarak devre dışı bırakıldı. Radar sadece
        # güncel/yaklaşan etkinlikleri listeliyor — geçmişe dönük bir arşivi
        # yok, bu yüzden geçmiş veri toplama amacıyla kullanılamıyor.
        # (fonksiyon kod tabanında duruyor, isterseniz ileride GÜNCEL/gelecek
        # etkinlik toplama işi için ayrı bir script'te kullanılabilir.)

        # 1. Setlist.fm (key varsa) — asıl geçmiş konser kaynağımız
        if args.sf_key:
            all_events += fetch_setlistfm(args.sf_key, args.start, args.end)
        else:
            print("[Setlist.fm] --sf-key verilmedi, atlandı.")

        # 2. Şehir Tiyatroları (URL/dosya varsa)
        if args.st_url:
            st_events = fetch_sehir_tiyatrolari(args.st_url, args.start, args.end)
            all_events += st_events
            if st_events:   # CSV okunamadıysa eski tiyatro kayıtları korunur
                yenilenen_kaynaklar.add("Şehir Tiyatroları")
        else:
            print("[Şehir Tiyatroları] --st-url / SEHIR_TIYATROLARI_CSV_URL boş, atlandı.")

        # 3. Mackolik — BJK ev sahibi maçları (her zaman, arsiv.mackolik.com)
        start_year = int(args.start[:4])
        end_year   = int(args.end[:4])
        all_events += scrape_bjk_fixtures(start_year, end_year)
        yenilenen_kaynaklar.add("Mackolik")

        # Mackolik sezon bazlı çalışıyor (örn. 2024/2025 sezonu Ağustos 2024 -
        # Mayıs 2025 arası sürer), bu yüzden istenen --start/--end aralığının
        # dışına taşan maçları burada kırpıyoruz.
        all_events = _filter_by_date(all_events, args.start, args.end)

    yeni_cekilen = len(all_events)

    # EKLEME MODU — mevcut kayıtlar önce gelir, deduplicate ilk görüleni
    # tuttuğu için aynı etkinlik tekrar çekilse de ESKİ kayıt korunur.
    mevcut = []
    if Path(args.out).exists() and not args.yenile:
        mevcut_df = pd.read_parquet(args.out)
        # Bu çalışmada yeniden çekilen kaynakların (Mackolik, Şehir Tiyatroları)
        # istenen aralıktaki eski kayıtları atılır — yoksa eski kayıtlar
        # (yanlış stadyum, sabit saat, silinmiş seanslar vb.) yenilerin
        # yerine korunurdu.
        if "source" in mevcut_df.columns:
            for kaynak in sorted(yenilenen_kaynaklar):
                tarih = mevcut_df["start_date_hour"].astype(str).str[:10]
                eski = ((mevcut_df["source"] == kaynak)
                        & (tarih >= args.start) & (tarih <= args.end))
                if kaynak == "Mackolik":
                    # Sezon sayfası alınamayan sezonların (Temmuz–Haziran) eski
                    # kayıtları korunur, yoksa o sezonun maçları kaybolurdu
                    for sezon in MACKOLIK_HATALI_SEZONLAR:
                        eski &= ~((tarih >= f"{sezon}-07-01") & (tarih <= f"{sezon + 1}-06-30"))
                if eski.any():
                    print(f"\n[Ekleme modu] {int(eski.sum())} eski {kaynak} kaydı yenileriyle değiştirilecek")
                mevcut_df = mevcut_df[~eski]
        mevcut    = mevcut_df.where(mevcut_df.notna(), None).to_dict("records")
        print(f"\n[Ekleme modu] {args.out} içinde {len(mevcut)} kayıt korunuyor")
    elif args.yenile:
        print(f"\n[--yenile] Mevcut {args.out} yok sayılıyor, sıfırdan yazılacak")

    # Yineleme temizle + tarihe göre sırala
    birlesik   = mevcut + all_events
    all_events = _konserleri_birlestir(deduplicate(birlesik))
    eklenen    = len(all_events) - len(mevcut)
    print(f"  Yeni çekilen: {yeni_cekilen} | eklenen: {eklenen} | "
          f"zaten vardı/yineleme: {len(birlesik) - len(all_events)}")
    all_events.sort(key=lambda e: e.get("start_datetime") or e.get("start_date_hour") or e.get("event_date") or "")

    df = pd.DataFrame(all_events)
    df.to_parquet(args.out, index=False, engine="pyarrow")
    df.head(1000).to_csv(
    args.out.replace(".parquet", "_ornek.csv"),
    index=False, encoding="utf-8-sig" )
    print(f"  Örnek CSV: {args.out.replace('.parquet', '_ornek.csv')}")

    # Kalite raporu
    kalite = {
        "kaynak":               "Setlist.fm + Şehir Tiyatroları + Mackolik",
        "toplam_etkinlik":      int(len(df)),
        "kaynak_dagilimi":      {k: int(v) for k, v in df["source"].value_counts().items()} if not df.empty else {},
        "kategori_dagilimi":    {k: int(v) for k, v in df["category"].value_counts().items()} if not df.empty else {},
        "onceki_kayit":         len(mevcut),
        "bu_calismada_eklenen": int(eklenen),
        "yineleme_silinen":     int(len(birlesik) - len(all_events)),
        "koordinatsiz_silinen": int(df[["lat","lon"]].isna().any(axis=1).sum()) if not df.empty else 0,
        "tarih_araligi":        {
            "baslangic": str(df["start_date_hour"].min())[:10] if not df.empty else args.start,
            "bitis":     str(df["start_date_hour"].max())[:10] if not df.empty else args.end,
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
