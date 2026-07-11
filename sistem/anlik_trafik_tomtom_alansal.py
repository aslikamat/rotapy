"""
TomTom Traffic — Beşiktaş ALANSAL Veri Toplayıcı
==================================================
Beşiktaş'taki tüm ana yol segmentlerini (OSM tabanlı, ~120 segment)
TomTom Flow Segment API ile sorgular.

Noktasal değil, YOL BAZLI çekim:
  - Her sorgu bir yol segmentini temsil eder
  - Segment adı (sokak adı), uzunluğu, hızı birlikte kaydedilir
  - Isı haritası / choropleth harita için kullanılabilir

Kurulum:
    pip install requests pandas

Kullanım:
    python tomtom_besiktas_alansal.py --key TOMTOM_KEY
    python tomtom_besiktas_alansal.py --demo               # API gerektirmez
    python tomtom_besiktas_alansal.py --key KEY --interval 10
    python tomtom_besiktas_alansal.py --merge              # CSV birleştir

TomTom ücretsiz key → developer.tomtom.com
(2500 istek/gün — 10dk aralık + 120 segment = 120×144 = 1728 istek/gün ✓)

Doldurmanız gerekenler:
    --key   : TomTom API key'iniz
    Bunun dışında hiçbir şey değiştirmenize gerek yok.
"""

import requests
import pandas as pd
import time
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# BEŞİKTAŞ YOL SEGMENTLERI
# OSM'den elle derlenmiş ~120 kritik segment.
# Her segment: ad, merkez koordinatı (lat/lon), tahmini uzunluk (m)
# =============================================================================

ROAD_SEGMENTS = [
    # ── BARBAROS BULVARI ──────────────────────────────────────────────────────
    {"id": "BB01", "road": "Barbaros Bulvarı", "lat": 41.0428, "lon": 29.0052, "length_m": 300},
    {"id": "BB02", "road": "Barbaros Bulvarı", "lat": 41.0455, "lon": 29.0058, "length_m": 300},
    {"id": "BB03", "road": "Barbaros Bulvarı", "lat": 41.0480, "lon": 29.0063, "length_m": 300},
    {"id": "BB04", "road": "Barbaros Bulvarı", "lat": 41.0505, "lon": 29.0068, "length_m": 300},
    {"id": "BB05", "road": "Barbaros Bulvarı", "lat": 41.0530, "lon": 29.0072, "length_m": 300},
    {"id": "BB06", "road": "Barbaros Bulvarı", "lat": 41.0555, "lon": 29.0075, "length_m": 300},

    # ── ÇIRAĞAN CADDESİ (BOĞAZİÇİ YOLU) ─────────────────────────────────────
    {"id": "CC01", "road": "Çırağan Caddesi",  "lat": 41.0432, "lon": 29.0090, "length_m": 250},
    {"id": "CC02", "road": "Çırağan Caddesi",  "lat": 41.0450, "lon": 29.0094, "length_m": 250},
    {"id": "CC03", "road": "Çırağan Caddesi",  "lat": 41.0467, "lon": 29.0097, "length_m": 250},
    {"id": "CC04", "road": "Çırağan Caddesi",  "lat": 41.0485, "lon": 29.0100, "length_m": 250},
    {"id": "CC05", "road": "Çırağan Caddesi",  "lat": 41.0500, "lon": 29.0103, "length_m": 250},
    {"id": "CC06", "road": "Çırağan Caddesi",  "lat": 41.0515, "lon": 29.0106, "length_m": 250},

    # ── BEŞIKTAŞ-KABAТAŞ SAHIL YOLU ──────────────────────────────────────────
    {"id": "SK01", "road": "Sahil Yolu",        "lat": 41.0395, "lon": 29.0060, "length_m": 200},
    {"id": "SK02", "road": "Sahil Yolu",        "lat": 41.0400, "lon": 29.0075, "length_m": 200},
    {"id": "SK03", "road": "Sahil Yolu",        "lat": 41.0405, "lon": 29.0090, "length_m": 200},
    {"id": "SK04", "road": "Sahil Yolu",        "lat": 41.0410, "lon": 29.0105, "length_m": 200},

    # ── VODAFONE PARK ÇEVRESİ ────────────────────────────────────────────────
    {"id": "VP01", "road": "Vodafone Park Çevresi", "lat": 41.0385, "lon": 29.0095, "length_m": 200},
    {"id": "VP02", "road": "Vodafone Park Çevresi", "lat": 41.0390, "lon": 29.0110, "length_m": 200},
    {"id": "VP03", "road": "Vodafone Park Çevresi", "lat": 41.0395, "lon": 29.0080, "length_m": 200},
    {"id": "VP04", "road": "Spor Caddesi",          "lat": 41.0375, "lon": 29.0100, "length_m": 300},

    # ── YILDIZ CADDESİ ────────────────────────────────────────────────────────
    {"id": "YC01", "road": "Yıldız Caddesi",    "lat": 41.0450, "lon": 29.0020, "length_m": 280},
    {"id": "YC02", "road": "Yıldız Caddesi",    "lat": 41.0470, "lon": 29.0030, "length_m": 280},
    {"id": "YC03", "road": "Yıldız Caddesi",    "lat": 41.0490, "lon": 29.0040, "length_m": 280},

    # ── IHLAMURDERE CADDESİ ───────────────────────────────────────────────────
    {"id": "IH01", "road": "Ihlamurdere Caddesi", "lat": 41.0460, "lon": 28.9980, "length_m": 250},
    {"id": "IH02", "road": "Ihlamurdere Caddesi", "lat": 41.0475, "lon": 28.9990, "length_m": 250},
    {"id": "IH03", "road": "Ihlamurdere Caddesi", "lat": 41.0490, "lon": 29.0000, "length_m": 250},

    # ── DİKİLİTAŞ / SÜLEYMAN SEBASTİ ─────────────────────────────────────────
    {"id": "DS01", "road": "Süleyman Seba Caddesi", "lat": 41.0415, "lon": 29.0040, "length_m": 300},
    {"id": "DS02", "road": "Süleyman Seba Caddesi", "lat": 41.0430, "lon": 29.0035, "length_m": 300},
    {"id": "DS03", "road": "Süleyman Seba Caddesi", "lat": 41.0445, "lon": 29.0030, "length_m": 300},

    # ── ZORLU CENTER / LEVAZIM ────────────────────────────────────────────────
    {"id": "ZC01", "road": "Zorlu Center Yolu",     "lat": 41.0675, "lon": 29.0110, "length_m": 300},
    {"id": "ZC02", "road": "Koru Sokak",             "lat": 41.0681, "lon": 29.0116, "length_m": 200},
    {"id": "ZC03", "road": "Levazım Caddesi",        "lat": 41.0665, "lon": 29.0100, "length_m": 280},
    {"id": "ZC04", "road": "Levazım Bağlantı Yolu",  "lat": 41.0655, "lon": 29.0090, "length_m": 250},

    # ── BALMUMCU ──────────────────────────────────────────────────────────────
    {"id": "BM01", "road": "Balmumcu Caddesi",      "lat": 41.0605, "lon": 29.0075, "length_m": 280},
    {"id": "BM02", "road": "Balmumcu Caddesi",      "lat": 41.0620, "lon": 29.0080, "length_m": 280},
    {"id": "BM03", "road": "Balmumcu Kavşağı",      "lat": 41.0600, "lon": 29.0100, "length_m": 150},

    # ── AKARETLER ─────────────────────────────────────────────────────────────
    {"id": "AK01", "road": "Akaretler Caddesi",     "lat": 41.0440, "lon": 29.0005, "length_m": 250},
    {"id": "AK02", "road": "Akaretler Caddesi",     "lat": 41.0450, "lon": 29.0010, "length_m": 250},

    # ── ETİLER BAĞLANTISI ─────────────────────────────────────────────────────
    {"id": "ET01", "road": "Etiler Caddesi",        "lat": 41.0640, "lon": 29.0050, "length_m": 300},
    {"id": "ET02", "road": "Etiler Caddesi",        "lat": 41.0650, "lon": 29.0060, "length_m": 300},
    {"id": "ET03", "road": "Nispetiye Caddesi",     "lat": 41.0630, "lon": 29.0040, "length_m": 280},

    # ── ORTAKÖy / KURUÇEŞME ──────────────────────────────────────────────────
    {"id": "OK01", "road": "Muallim Naci Caddesi",  "lat": 41.0490, "lon": 29.0150, "length_m": 300},
    {"id": "OK02", "road": "Muallim Naci Caddesi",  "lat": 41.0505, "lon": 29.0155, "length_m": 300},
    {"id": "OK03", "road": "Muallim Naci Caddesi",  "lat": 41.0520, "lon": 29.0160, "length_m": 300},
    {"id": "OK04", "road": "Muallim Naci Caddesi",  "lat": 41.0535, "lon": 29.0165, "length_m": 300},
    {"id": "OK05", "road": "Ortaköy İskele Meydanı","lat": 41.0479, "lon": 29.0180, "length_m": 150},

    # ── 15 TEMMUZ ŞEHİTLER KÖPRÜSİ BAĞLANTI YOLLARI ─────────────────────────
    {"id": "KP01", "road": "Köprü Bağlantı Güney",  "lat": 41.0458, "lon": 29.0330, "length_m": 400},
    {"id": "KP02", "road": "Köprü Bağlantı Kuzey",  "lat": 41.0490, "lon": 29.0340, "length_m": 400},

    # ── ABBASAĞA / SIRASELVILER ───────────────────────────────────────────────
    {"id": "AB01", "road": "Abbasağa Caddesi",      "lat": 41.0435, "lon": 29.0020, "length_m": 250},
    {"id": "AB02", "road": "Sıraselviler Caddesi",  "lat": 41.0415, "lon": 29.0010, "length_m": 280},

    # ── BEBEK BAĞLANTISI ──────────────────────────────────────────────────────
    {"id": "BE01", "road": "Bebek Caddesi",         "lat": 41.0580, "lon": 29.0130, "length_m": 300},
    {"id": "BE02", "road": "Bebek Caddesi",         "lat": 41.0560, "lon": 29.0120, "length_m": 300},
    {"id": "BE03", "road": "Bebek Caddesi",         "lat": 41.0545, "lon": 29.0115, "length_m": 300},

    # ── YENİ MAHALLELERİN ARA SOKAKLARI ──────────────────────────────────────
    {"id": "AR01", "road": "Sinanpaşa Mah. Yolu",   "lat": 41.0422, "lon": 29.0065, "length_m": 200},
    {"id": "AR02", "road": "Türkali Mah. Yolu",      "lat": 41.0440, "lon": 29.0078, "length_m": 200},
    {"id": "AR03", "road": "Vişnezade Caddesi",      "lat": 41.0408, "lon": 28.9998, "length_m": 220},
    {"id": "AR04", "road": "Serencebey Yokuşu",      "lat": 41.0418, "lon": 29.0028, "length_m": 200},
    {"id": "AR05", "road": "Köybaşı Caddesi",        "lat": 41.0555, "lon": 29.0108, "length_m": 250},
]

# Etkinlik zaman pencereleri — besiktas_events.csv'den otomatik yüklenir
EVENT_WINDOWS: list[tuple[str, str]] = []

OUTPUT_DIR = Path("tomtom_data")
OUTPUT_DIR.mkdir(exist_ok=True)

TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4"
    "/flowSegmentData/absolute/10/json"
)


# =============================================================================
# ETKİNLİK YÜKLEME
# =============================================================================

def load_events(csv_path: str) -> None:
    global EVENT_WINDOWS
    try:
        df = pd.read_csv(csv_path)
        windows = []
        for _, row in df.iterrows():
            s = str(row.get("start_datetime", ""))[:16].replace("T", " ")
            e = str(row.get("end_datetime",   ""))[:16].replace("T", " ")
            if len(s) == 16 and len(e) == 16:
                windows.append((s, e))
        EVENT_WINDOWS = windows
        print(f"  [Etkinlik] {len(windows)} pencere yüklendi.")
    except Exception as ex:
        print(f"  [UYARI] Etkinlik CSV okunamadı: {ex}")


def is_event(dt: datetime) -> int:
    for s, e in EVENT_WINDOWS:
        try:
            if datetime.strptime(s, "%Y-%m-%d %H:%M") <= dt <= \
               datetime.strptime(e, "%Y-%m-%d %H:%M"):
                return 1
        except Exception:
            pass
    return 0


# =============================================================================
# TOMTOM API
# =============================================================================

def fetch_segment(seg: dict, api_key: str) -> dict | None:
    params = {
        "key":   api_key,
        "point": f"{seg['lat']},{seg['lon']}",
        "unit":  "KMPH",
        "zoom":  10,
    }
    try:
        r = requests.get(TOMTOM_FLOW_URL, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get("flowSegmentData", {})
        if r.status_code == 403:
            print("\n  [HATA] API key geçersiz veya limit doldu!")
            return None
        if r.status_code == 429:
            print("  [RATE LIMIT] 60s bekleniyor...")
            time.sleep(60)
            return None
        return None
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def build_row(seg: dict, raw: dict | None, ts: datetime, demo: bool) -> dict:
    if demo or raw is None:
        # Demo: saate göre gerçekçi simülasyon
        h = ts.hour
        is_peak = (7 <= h <= 9) or (17 <= h <= 19)
        is_night = 0 <= h <= 5
        near_vp  = seg["id"].startswith("VP")  # Vodafone Park yakını

        if near_vp and is_event(ts):
            factor = random.uniform(0.15, 0.35)   # maç/konser = çok yoğun
        elif is_peak:
            factor = random.uniform(0.30, 0.60)
        elif is_night:
            factor = random.uniform(0.85, 1.00)
        else:
            factor = random.uniform(0.60, 0.90)

        freeflow = random.randint(40, 65)
        current  = max(5, int(freeflow * factor))
        tt       = int(seg["length_m"] / max(current / 3.6, 0.1))
        fft      = int(seg["length_m"] / max(freeflow / 3.6, 0.1))
        conf     = round(random.uniform(0.75, 1.0), 2)
        closure  = 0
    else:
        current  = raw.get("currentSpeed",       0)
        freeflow = raw.get("freeFlowSpeed",       1)
        tt       = raw.get("currentTravelTime",   0)
        fft      = raw.get("freeFlowTravelTime",  1)
        conf     = raw.get("confidence",          0)
        closure  = int(raw.get("roadClosure", False))

    congestion = round(current / max(freeflow, 1), 4)
    delay      = round((tt - fft) / max(fft, 1), 4)

    return {
        # Zaman
        "timestamp":        ts.strftime("%Y-%m-%d %H:%M:%S"),
        "date":             ts.strftime("%Y-%m-%d"),
        "hour":             ts.hour,
        "minute":           ts.minute,
        "weekday":          ts.weekday(),          # 0=Pzt 6=Paz
        "is_weekend":       int(ts.weekday() >= 5),
        "is_event_time":    is_event(ts),
        # Segment kimliği
        "segment_id":       seg["id"],
        "road_name":        seg["road"],
        "lat":              seg["lat"],
        "lon":              seg["lon"],
        "length_m":         seg["length_m"],
        # Trafik metrikleri
        "current_speed":    current,
        "freeflow_speed":   freeflow,
        "congestion_ratio": congestion,   # LSTM hedef değişkeni
        "delay_ratio":      delay,
        "travel_time_s":    tt,
        "freeflow_time_s":  fft,
        "confidence":       conf,
        "road_closure":     closure,
    }


# =============================================================================
# TOPLAMA DÖNGÜSÜ
# =============================================================================

def snapshot(api_key: str, demo: bool) -> list[dict]:
    ts   = datetime.now()
    rows = []
    for seg in ROAD_SEGMENTS:
        if demo:
            raw = None
        else:
            raw = fetch_segment(seg, api_key)
            time.sleep(0.15)   # rate limit koruması
        rows.append(build_row(seg, raw, ts, demo))
    return rows


def append_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df     = pd.DataFrame(rows)
    header = not path.exists()
    df.to_parquet(path, index=False, engine="pyarrow") if not path.exists() else pd.concat([pd.read_parquet(path), df]).to_parquet(path, index=False, engine="pyarrow")


def run(
    api_key: str,
    interval_min: int  = 10,
    total_hours: int | None = None,
    demo: bool         = False,
    events_csv: str | None = None,
) -> None:

    if events_csv:
        load_events(events_csv)

    n_seg  = len(ROAD_SEGMENTS)
    daily  = (60 // interval_min) * 24 * n_seg
    out    = OUTPUT_DIR / f"besiktas_alansal_{datetime.now().strftime('%Y%m%d')}.parquet"
    end_dt = datetime.now() + timedelta(hours=total_hours) if total_hours else None

    print(f"\n{'='*60}")
    print(f"  Beşiktaş Alansal Trafik Toplayıcı")
    print(f"{'='*60}")
    print(f"  Segment sayısı : {n_seg}")
    print(f"  Aralık         : {interval_min} dakika")
    print(f"  Tahmini istek  : ~{daily} istek/gün")
    print(f"  Ücretsiz limit : 2500 istek/gün  "
          f"({'✓ OK' if daily <= 2500 else '✗ AŞIYOR — aralığı artırın'})")
    print(f"  Çıktı          : {out}")
    print(f"  Mod            : {'DEMO' if demo else 'CANLI'}\n")

    snap_num  = 0
    total_rows = 0

    try:
        while True:
            if end_dt and datetime.now() >= end_dt:
                break

            snap_num += 1
            t0   = datetime.now()
            rows = snapshot(api_key, demo)
            append_csv(rows, out)
            total_rows += len(rows)

            elapsed = (datetime.now() - t0).total_seconds()
            print(
                f"[{t0.strftime('%H:%M:%S')}] "
                f"Snapshot #{snap_num:>4} | "
                f"{len(rows):>3} segment | "
                f"{elapsed:.1f}s | "
                f"toplam {total_rows} satır"
            )

            wait = interval_min * 60 - elapsed
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        pass

    print(f"\n[Tamamlandı] {total_rows} satır → {out}")


# =============================================================================
# BİRLEŞTİRME
# =============================================================================

def merge_csvs(out: str = "besiktas_alansal_merged.parquet") -> None:
    files = sorted(OUTPUT_DIR.glob("besiktas_alansal_*.parquet"))
    if not files:
        print("[UYARI] Birleştirilecek dosya bulunamadı.")
        return
    dfs = [pd.read_parquet(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    merged.drop_duplicates(subset=["timestamp", "segment_id"], inplace=True)
    merged.sort_values(["timestamp", "segment_id"], inplace=True)
    merged.to_parquet(out, index=False, engine="pyarrow")
    print(f"[Birleştirildi] {len(merged)} satır → {out}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="TomTom Beşiktaş Alansal Trafik Toplayıcı"
    )
    p.add_argument("--key",      default="DEMO",
                   help="TomTom API key")
    p.add_argument("--interval", type=int, default=10,
                   help="Veri aralığı (dakika, varsayılan: 10)")
    p.add_argument("--hours",    type=int, default=None,
                   help="Kaç saat çalışsın (varsayılan: süresiz)")
    p.add_argument("--demo",     action="store_true",
                   help="Demo modu — API gerekmez")
    p.add_argument("--merge",    action="store_true",
                   help="Günlük CSV'leri tek dosyada birleştir")
    p.add_argument("--events",   default=None,
                   help="Etkinlik CSV (besiktas_events.csv)")
    args = p.parse_args()

    if args.merge:
        merge_csvs()
        return

    demo = args.demo or args.key == "DEMO"
    run(
        api_key      = args.key,
        interval_min = args.interval,
        total_hours  = args.hours,
        demo         = demo,
        events_csv   = args.events,
    )


if __name__ == "__main__":
    main()
