"""
TomTom Traffic — Beşiktaş ALANSAL Veri Toplayıcı
==================================================
OSM_onemli_yollar.py'nin seçtiği yol noktalarını (tomtom_noktalar.json)
TomTom Flow Segment API ile sorgular.

Nasıl çalışır:
  - TomTom'a her noktanın lat/lon'u gönderilir; TomTom noktaya en yakın yol
    parçasının anlık hızını, serbest akış hızını ve geometrisini döndürür.
  - Her ölçüm AYRI satır olarak kaydedilir (yol bazlı ham kayıt). İleride
    rota önerisi için "yol, hücre ortalamasına göre nasıl davranıyor"
    bilgisini çıkarmak bu ham kayda dayanır — hücre ortalaması ayrıca üretilir.
  - geohash sütunu noktanın kendi geohash'i DEĞİL, noktanın düştüğü İBB
    hücresinin eğitim verisindeki geohash'idir (ör. sxk9kpe). Bu değer
    tomtom_noktalar.json'dan gelir, burada hesaplanmaz.

Kurulum:
    pip install requests pandas pyarrow

Kullanım:
    python anlik_trafik_tomtom_alansal.py                  # canlı, 60 dk aralık
    python anlik_trafik_tomtom_alansal.py --demo           # API gerektirmez (simülasyon)
    python anlik_trafik_tomtom_alansal.py --hours 3        # 3 saat çalış, dur
    python anlik_trafik_tomtom_alansal.py --merge          # günlük parquet'leri birleştir

Girdi:
    tomtom_noktalar.json  (script'in yanında; OSM_onemli_yollar.py üretir)

Çıktı:
    tomtom_data/besiktas_alansal_YYYYMMDD.parquet   (script'in yanında)
    besiktas_alansal_merged.parquet                 (--merge ile)

Kota: ücretsiz katman ~2.500 istek/gün. İstek sayısı = nokta sayısı × günlük
sorgu sayısı; program başlarken hesaplanıp ekrana yazılır.

Başarısız istekler: değerler boş (NaN) bırakılır ve ok=0 yazılır — sahte/simüle
veri ASLA gerçek veriyle karışmaz. Simülasyon sadece --demo ile çalışır.
"""

import argparse
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# =============================================================================
# DOSYALAR VE SABİTLER
# =============================================================================

SCRIPT_DIR    = Path(__file__).resolve().parent
NOKTA_DOSYASI = SCRIPT_DIR / "tomtom_noktalar.json"
OUTPUT_DIR    = SCRIPT_DIR / "tomtom_data"
MERGED_PATH   = SCRIPT_DIR / "besiktas_alansal_merged.parquet"

# Adresteki sayı ZOOM seviyesidir (0-22). TomTom noktayı bu zoom'da görünen en
# yakın yola bağlar: zoom 10'da sadece büyük arterler görünür ve ara yollardaki
# noktalar yüzlerce metre ötedeki artere bağlanır. 22 = en ayrıntılı seviye,
# her nokta kendi yoluna bağlanır.
TOMTOM_ZOOM = 22
TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4"
    f"/flowSegmentData/absolute/{TOMTOM_ZOOM}/json"
)

# Tüpraş Stadyumu'nun bulunduğu İBB hücresi — demo modunda etkinlik etkisi burada
STADYUM_HUCRESI = "sxk97z"

GUNLUK_KOTA = 2500

# Etkinlik zaman pencereleri — --events ile verilen CSV'den yüklenir
EVENT_WINDOWS: list[tuple[str, str]] = []


# =============================================================================
# NOKTALAR
# =============================================================================

def load_points(path: Path = NOKTA_DOSYASI) -> list[dict]:
    """
    OSM_onemli_yollar.py'nin ürettiği nokta listesini okur.
    Dosya yoksa ya da beklenen alanlar eksikse hata verir — eski/sabit bir
    listeye sessizce geçmez.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Nokta dosyası bulunamadı: {path}\n"
            f"Önce OSM_onemli_yollar.py'yi çalıştırın.")
    points = json.loads(path.read_text(encoding="utf-8"))
    gerekli = {"id", "road", "lat", "lon", "cell", "geohash"}
    for p in points:
        eksik = gerekli - p.keys()
        if eksik:
            raise ValueError(f"{path} içinde eksik alan {sorted(eksik)}: {p}")
    ids = [p["id"] for p in points]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} içinde tekrar eden id var")
    if not points:
        raise ValueError(f"{path} boş")
    return points


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
    """Başarılıysa flowSegmentData sözlüğü, değilse None döner."""
    params = {
        "key":   api_key,
        "point": f"{seg['lat']},{seg['lon']}",
        "unit":  "KMPH",
    }
    try:
        r = requests.get(TOMTOM_FLOW_URL, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get("flowSegmentData") or None
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


def _segment_coords(raw: dict) -> str | None:
    """TomTom'un ölçtüğü yol parçasının geometrisi → '[[lat,lon],...]' metni."""
    pts = (raw.get("coordinates") or {}).get("coordinate") or []
    if not pts:
        return None
    return json.dumps([[round(p["latitude"], 6), round(p["longitude"], 6)] for p in pts],
                      separators=(",", ":"))


def _sayi(x):
    return float(x) if isinstance(x, (int, float)) else float("nan")


def build_row(seg: dict, raw: dict | None, ts: datetime, demo: bool) -> dict:
    nan = float("nan")
    frc, coords = None, None

    if demo:
        # Demo: saate göre simülasyon (sadece --demo ile; gerçek veriye karışmaz)
        h = ts.hour
        is_peak  = (7 <= h <= 9) or (17 <= h <= 19)
        is_night = 0 <= h <= 5
        stadyum  = seg["cell"] == STADYUM_HUCRESI

        if stadyum and is_event(ts):
            factor = random.uniform(0.15, 0.35)   # maç/konser = çok yoğun
        elif is_peak:
            factor = random.uniform(0.30, 0.60)
        elif is_night:
            factor = random.uniform(0.85, 1.00)
        else:
            factor = random.uniform(0.60, 0.90)

        length   = seg.get("length_m", 300)
        freeflow = float(random.randint(40, 65))
        current  = float(max(5, int(freeflow * factor)))
        tt       = float(int(length / max(current / 3.6, 0.1)))
        fft      = float(int(length / max(freeflow / 3.6, 0.1)))
        conf     = round(random.uniform(0.75, 1.0), 2)
        closure  = 0
        ok       = 1
    elif raw:
        current  = _sayi(raw.get("currentSpeed"))
        freeflow = _sayi(raw.get("freeFlowSpeed"))
        tt       = _sayi(raw.get("currentTravelTime"))
        fft      = _sayi(raw.get("freeFlowTravelTime"))
        conf     = _sayi(raw.get("confidence"))
        closure  = int(bool(raw.get("roadClosure", False)))
        frc      = raw.get("frc")
        coords   = _segment_coords(raw)
        ok       = 1
    else:
        # İstek başarısız: değerler boş, ok=0 — simülasyonla DOLDURULMAZ
        current = freeflow = tt = fft = conf = nan
        closure = None
        ok      = 0

    if ok and freeflow == freeflow and freeflow > 0 and current == current:
        congestion = round(min(max(1 - current / freeflow, 0), 1), 4)
    else:
        congestion = nan
    if ok and fft == fft and fft > 0 and tt == tt:
        delay = round((tt - fft) / fft, 4)
    else:
        delay = nan

    return {
        # Zaman
        "timestamp":        ts.strftime("%Y-%m-%d %H:%M:%S"),
        "date_hour":        ts.strftime("%Y-%m-%d %H"),
        "weekday":          ts.weekday(),          # 0=Pzt 6=Paz
        "is_weekend":       int(ts.weekday() >= 5),
        "is_event_time":    is_event(ts),
        # Nokta (tomtom_noktalar.json)
        "id":               seg["id"],
        "road":             seg["road"],
        "highway":          seg.get("highway"),
        "secim":            seg.get("secim"),
        "cell":             seg["cell"],
        "geohash":          seg["geohash"],        # eğitim verisindeki hücre geohash'i
        "lat":              seg["lat"],
        "lon":              seg["lon"],
        "length_m":         seg.get("length_m"),
        # Durum
        "ok":               ok,                    # 1 = ölçüm var, 0 = istek başarısız
        "demo":             int(demo),
        # Trafik metrikleri
        "current_speed":    current,
        "freeflow_speed":   freeflow,
        "congestion_ratio": congestion,
        "delay_ratio":      delay,
        "travel_time_s":    tt,
        "freeflow_time_s":  fft,
        "confidence":       conf,
        "road_closure":     closure,
        # TomTom'un gerçekte ölçtüğü yol parçası
        "frc":              frc,                   # TomTom yol sınıfı (FRC0 = otoyol ...)
        "segment_coords":   coords,                # '[[lat,lon],...]'
    }


# =============================================================================
# TOPLAMA DÖNGÜSÜ
# =============================================================================

def snapshot(points: list[dict], api_key: str, demo: bool) -> list[dict]:
    ts   = datetime.now()
    rows = []
    for seg in points:
        if demo:
            raw = None
        else:
            raw = fetch_segment(seg, api_key)
            time.sleep(0.15)   # rate limit koruması
        rows.append(build_row(seg, raw, ts, demo))
    return rows


def append_parquet(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def run(
    api_key: str,
    points: list[dict],
    interval_min: int  = 60,
    total_hours: int | None = None,
    demo: bool         = False,
    events_csv: str | None = None,
) -> None:

    if events_csv:
        load_events(events_csv)

    OUTPUT_DIR.mkdir(exist_ok=True)

    n_seg   = len(points)
    n_cell  = len({p["geohash"] for p in points})
    daily   = int(1440 / interval_min) * n_seg
    end_dt  = datetime.now() + timedelta(hours=total_hours) if total_hours else None

    print(f"\n{'='*60}")
    print(f"  Beşiktaş Alansal Trafik Toplayıcı")
    print(f"{'='*60}")
    print(f"  Nokta dosyası  : {NOKTA_DOSYASI}")
    print(f"  Nokta sayısı   : {n_seg}  ({n_cell} hücre)")
    print(f"  Aralık         : {interval_min} dakika")
    print(f"  Tahmini istek  : ~{daily} istek/gün")
    print(f"  Ücretsiz limit : {GUNLUK_KOTA} istek/gün  "
          f"({'✓ OK' if daily <= GUNLUK_KOTA else '✗ AŞIYOR — aralığı artırın'})")
    print(f"  Çıktı klasörü  : {OUTPUT_DIR}")
    print(f"  Mod            : {'DEMO' if demo else 'CANLI'}\n")

    snap_num   = 0
    total_rows = 0

    try:
        while True:
            if end_dt and datetime.now() >= end_dt:
                break

            snap_num += 1
            t0   = datetime.now()
            out  = OUTPUT_DIR / f"besiktas_alansal_{t0.strftime('%Y%m%d')}.parquet"
            rows = snapshot(points, api_key, demo)
            append_parquet(rows, out)
            total_rows += len(rows)
            basarisiz = sum(1 for r in rows if not r["ok"])

            elapsed = (datetime.now() - t0).total_seconds()
            print(
                f"[{t0.strftime('%H:%M:%S')}] "
                f"Snapshot #{snap_num:>4} | "
                f"{len(rows):>3} nokta | "
                f"{basarisiz:>3} başarısız | "
                f"{elapsed:.1f}s | "
                f"toplam {total_rows} satır"
            )

            wait = interval_min * 60 - elapsed
            if wait > 0:
                if end_dt:
                    wait = min(wait, max((end_dt - datetime.now()).total_seconds(), 0))
                time.sleep(wait)

    except KeyboardInterrupt:
        pass

    print(f"\n[Tamamlandı] {total_rows} satır → {OUTPUT_DIR}")


# =============================================================================
# BİRLEŞTİRME
# =============================================================================

def merge_files(out: Path = MERGED_PATH) -> None:
    files = sorted(OUTPUT_DIR.glob("besiktas_alansal_[0-9]*.parquet"))
    if not files:
        print(f"[UYARI] {OUTPUT_DIR} içinde birleştirilecek dosya bulunamadı.")
        return
    merged = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # Aynı hücrede birden fazla nokta var → tekrar kontrolü nokta id'si ile
    merged.drop_duplicates(subset=["timestamp", "id"], inplace=True)
    merged.sort_values(["timestamp", "id"], inplace=True)
    merged.to_parquet(out, index=False, engine="pyarrow")
    print(f"[Birleştirildi] {len(files)} dosya, {len(merged)} satır → {out}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="TomTom Beşiktaş Alansal Trafik Toplayıcı"
    )
    p.add_argument("--key",      default="9dqoh1Ho4b8xTvYB3tnZCtGEb5qUS2S1",
                   help="TomTom API key")
    p.add_argument("--interval", type=int, default=60,
                   help="Veri aralığı (dakika, varsayılan: 60)")
    p.add_argument("--hours",    type=int, default=None,
                   help="Kaç saat çalışsın (varsayılan: süresiz)")
    p.add_argument("--demo",     action="store_true",
                   help="Demo modu — API gerekmez, simüle veri üretir")
    p.add_argument("--merge",    action="store_true",
                   help="Günlük parquet dosyalarını tek dosyada birleştir")
    p.add_argument("--events",   default=None,
                   help="Etkinlik CSV (start_datetime / end_datetime sütunları)")
    args = p.parse_args()

    if args.merge:
        merge_files()
        return

    demo = args.demo or args.key == "DEMO"
    run(
        api_key      = args.key,
        points       = load_points(),
        interval_min = args.interval,
        total_hours  = args.hours,
        demo         = demo,
        events_csv   = args.events,
    )


if __name__ == "__main__":
    main()
