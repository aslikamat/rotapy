"""
Open-Meteo Hava Durumu Verisi — Beşiktaş (GEÇMİŞ, GEOHASH BAZLI, ÇOKLU NOKTA)
================================================================================
2020-01-01 → 2024-12-31 arası saatlik hava verisi.
Endpoint: archive-api.open-meteo.com
LSTM model eğitimi için.

Kayıt gerekmez, tamamen ücretsiz.

Kurulum:
    pip install requests pandas pyarrow

Kullanım:
    python gecmis_hava_durumu.py
    python gecmis_hava_durumu.py --start 2022-01-01 --end 2023-12-31

Çıktı:
    gecmis_hava_durumu.parquet  → LSTM eğitim verisi

Sütunlar:
    timestamp, date_hour, weekday, is_weekend
    lat, lon, geohash   → trafik/etkinlik scriptleriyle aynı anahtar,
                           birleştirme (join) için kullanılır
    temperature_c       → Sıcaklık (°C)
    precipitation_mm    → Yağış miktarı (mm)
    wind_speed_kmh      → Rüzgar hızı (km/h)
    cloud_cover_pct     → Bulut örtüsü (%)
    humidity_pct        → Nem (%)
    weather_code        → WMO hava kodu
    is_rainy            → Yağmurlu mu? (1/0)
    is_snowy            → Karlı mı? (1/0)
    is_stormy           → Fırtınalı mı? (1/0)
    is_bad_weather      → Zorlu hava koşulu mu? (1/0) → ana LSTM feature

Anlık versiyonla fark:
    Aynı 58 nokta, aynı geohash, aynı sütun isimleri kullanılıyor.
    Birleştirme (join) scriptinde her iki versiyon tutarlı şekilde
    eşleşir — geohash + date_hour anahtarıyla.

Veri hacmi: 5 yıl × 58 nokta × saatlik ≈ 2.5 milyon satır.
Gerekirse --start/--end ile aralığı daraltın.
"""

import requests
import pandas as pd
import argparse
import json
from datetime import datetime
from pathlib import Path

# =============================================================================
# BEŞİKTAŞ NOKTALARI
# anlik_hava_durumu.py'deki AYNI 58 koordinat — join tutarlılığı için
# =============================================================================

POINTS = [
    {"lat": 41.0428, "lon": 29.0052},
    {"lat": 41.0455, "lon": 29.0058},
    {"lat": 41.048,  "lon": 29.0063},
    {"lat": 41.0505, "lon": 29.0068},
    {"lat": 41.053,  "lon": 29.0072},
    {"lat": 41.0555, "lon": 29.0075},
    {"lat": 41.0432, "lon": 29.009},
    {"lat": 41.045,  "lon": 29.0094},
    {"lat": 41.0467, "lon": 29.0097},
    {"lat": 41.0485, "lon": 29.01},
    {"lat": 41.05,   "lon": 29.0103},
    {"lat": 41.0515, "lon": 29.0106},
    {"lat": 41.0395, "lon": 29.006},
    {"lat": 41.04,   "lon": 29.0075},
    {"lat": 41.0405, "lon": 29.009},
    {"lat": 41.041,  "lon": 29.0105},
    {"lat": 41.0385, "lon": 29.0095},
    {"lat": 41.039,  "lon": 29.011},
    {"lat": 41.0395, "lon": 29.008},
    {"lat": 41.0375, "lon": 29.01},
    {"lat": 41.045,  "lon": 29.002},
    {"lat": 41.047,  "lon": 29.003},
    {"lat": 41.049,  "lon": 29.004},
    {"lat": 41.046,  "lon": 28.998},
    {"lat": 41.0475, "lon": 28.999},
    {"lat": 41.049,  "lon": 29.0},
    {"lat": 41.0415, "lon": 29.004},
    {"lat": 41.043,  "lon": 29.0035},
    {"lat": 41.0445, "lon": 29.003},
    {"lat": 41.0675, "lon": 29.011},
    {"lat": 41.0681, "lon": 29.0116},
    {"lat": 41.0665, "lon": 29.01},
    {"lat": 41.0655, "lon": 29.009},
    {"lat": 41.0605, "lon": 29.0075},
    {"lat": 41.062,  "lon": 29.008},
    {"lat": 41.06,   "lon": 29.01},
    {"lat": 41.044,  "lon": 29.0005},
    {"lat": 41.045,  "lon": 29.001},
    {"lat": 41.064,  "lon": 29.005},
    {"lat": 41.065,  "lon": 29.006},
    {"lat": 41.063,  "lon": 29.004},
    {"lat": 41.049,  "lon": 29.015},
    {"lat": 41.0505, "lon": 29.0155},
    {"lat": 41.052,  "lon": 29.016},
    {"lat": 41.0535, "lon": 29.0165},
    {"lat": 41.0479, "lon": 29.018},
    {"lat": 41.0458, "lon": 29.033},
    {"lat": 41.049,  "lon": 29.034},
    {"lat": 41.0435, "lon": 29.002},
    {"lat": 41.0415, "lon": 29.001},
    {"lat": 41.058,  "lon": 29.013},
    {"lat": 41.056,  "lon": 29.012},
    {"lat": 41.0545, "lon": 29.0115},
    {"lat": 41.0422, "lon": 29.0065},
    {"lat": 41.044,  "lon": 29.0078},
    {"lat": 41.0408, "lon": 28.9998},
    {"lat": 41.0418, "lon": 29.0028},
    {"lat": 41.0555, "lon": 29.0108},
]

TIMEZONE        = "Europe/Istanbul"
GEOHASH_PRECISION = 7

# =============================================================================
# GEOHASH — harici kütüphane gerektirmeyen saf Python implementasyonu
# =============================================================================

_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def encode_geohash(lat: float, lon: float, precision: int = GEOHASH_PRECISION) -> str:
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    geohash   = []
    bits      = [16, 8, 4, 2, 1]
    bit, ch, even = 0, 0, True
    while len(geohash) < precision:
        if even:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon > mid:
                ch |= bits[bit]; lon_range[0] = mid
            else:
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat > mid:
                ch |= bits[bit]; lat_range[0] = mid
            else:
                lat_range[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(_GEOHASH_BASE32[ch])
            bit = ch = 0
    return "".join(geohash)


# Her noktanın geohash'ini önceden hesapla
for _p in POINTS:
    _p["geohash"] = encode_geohash(_p["lat"], _p["lon"], GEOHASH_PRECISION)

# =============================================================================
# WMO HAVA KODU SINIFLANDIRMASI
# =============================================================================

def classify_weather(code: int) -> dict:
    """
    WMO kodunu trafik etkisi açısından sınıflandırır.

    0-3   : Açık / az bulutlu    → normal trafik
    45-48 : Sis                  → yavaşlama
    51-67 : Yağmur               → yoğunlaşma
    71-77 : Kar                  → ciddi yavaşlama
    80-82 : Sağanak              → yoğunlaşma
    85-86 : Kar sağanağı         → ciddi yavaşlama
    95-99 : Fırtına              → çok ciddi yavaşlama
    """
    return {
        "is_rainy":       int(51 <= code <= 67 or 80 <= code <= 82),
        "is_snowy":       int(71 <= code <= 77 or 85 <= code <= 86),
        "is_stormy":      int(95 <= code <= 99),
        "is_foggy":       int(45 <= code <= 48),
        "is_bad_weather": int(code >= 45),
    }


# =============================================================================
# YARDIMCI — çoklu konum API parametreleri
# Open-Meteo, virgülle ayrılmış lat/lon listesi kabul ediyor
# =============================================================================

HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "relative_humidity_2m",
    "weather_code",
    "snow_depth",
]


def _multi_location_params(extra: dict) -> dict:
    """58 noktanın tamamı için tek API isteği parametresi oluşturur."""
    params = {
        "latitude":        ",".join(str(p["lat"]) for p in POINTS),
        "longitude":       ",".join(str(p["lon"]) for p in POINTS),
        "hourly":          ",".join(HOURLY_VARS),
        "timezone":        TIMEZONE,
        "wind_speed_unit": "kmh",
    }
    params.update(extra)
    return params


def _parse_multi_location_response(data) -> pd.DataFrame:
    """
    Open-Meteo çoklu konum yanıtını düz tabloya dönüştürür.
    API, her nokta için ayrı bir dict içeren liste döndürür.
    """
    if isinstance(data, dict):
        data = [data]   # tek nokta yanıtı

    frames = []
    for i, item in enumerate(data):
        hourly = item.get("hourly", {})
        df     = pd.DataFrame(hourly)
        df["lat"]     = POINTS[i]["lat"]
        df["lon"]     = POINTS[i]["lon"]
        df["geohash"] = POINTS[i]["geohash"]
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =============================================================================
# GEÇMİŞ VERİ — archive-api.open-meteo.com
# =============================================================================

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_historical(
    start_date: str = "2020-01-01",
    end_date:   str = "2024-12-31",
) -> pd.DataFrame:
    """
    Belirtilen tarih aralığı için 58 noktanın saatlik geçmiş
    hava verisini çeker. Yıllık parçalar halinde indirilir.
    """
    print(f"\n[Geçmiş Hava] {start_date} → {end_date} çekiliyor "
          f"({len(POINTS)} nokta/geohash)...")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    chunks  = []
    current = start
    while current <= end:
        chunk_end = min(datetime(current.year, 12, 31), end)
        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))
        current = datetime(current.year + 1, 1, 1)

    all_dfs = []
    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] {chunk_start} → {chunk_end}...", end=" ")
        params = _multi_location_params({
            "start_date": chunk_start,
            "end_date":   chunk_end,
        })
        try:
            r  = requests.get(HISTORICAL_URL, params=params, timeout=60)
            r.raise_for_status()
            df = _parse_multi_location_response(r.json())
            all_dfs.append(df)
            print(f"{len(df):,} satır ✓")
        except Exception as e:
            print(f"[HATA] {e}")

    if not all_dfs:
        return pd.DataFrame()

    merged = pd.concat(all_dfs, ignore_index=True)
    return _process(merged)


# =============================================================================
# ORTAK İŞLEME — anlik_hava_durumu.py ile AYNI mantık
# =============================================================================

def _process(df: pd.DataFrame) -> pd.DataFrame:
    """Ham API çıktısını LSTM'e hazır formata dönüştürür."""
    if df.empty:
        return df

    df = df.rename(columns={
        "time":                 "timestamp",
        "temperature_2m":       "temperature_c",
        "precipitation":        "precipitation_mm",
        "wind_speed_10m":       "wind_speed_kmh",
        "cloud_cover":          "cloud_cover_pct",
        "relative_humidity_2m": "humidity_pct",
        "weather_code":         "weather_code",
        "snow_depth":           "snow_depth_m",
    })

    # Zaman özellikleri — anlik_hava_durumu.py ile AYNI sütun isimleri
    df["timestamp"]  = pd.to_datetime(df["timestamp"])
    df["date_hour"]  = df["timestamp"].dt.strftime("%Y-%m-%d %H")
    df["weekday"]    = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Hava kodu sınıflandırması
    df["weather_code"] = df["weather_code"].fillna(0).astype(int)
    weather_flags = df["weather_code"].apply(classify_weather).apply(pd.Series)
    df = pd.concat([df, weather_flags], axis=1)

    # Sütun sırası — anlik_hava_durumu.py ile AYNI
    cols = [
        "timestamp", "date_hour", "weekday", "is_weekend",
        "lat", "lon", "geohash",
        "temperature_c", "precipitation_mm", "wind_speed_kmh",
        "cloud_cover_pct", "humidity_pct", "snow_depth_m",
        "weather_code",
        "is_rainy", "is_snowy", "is_stormy", "is_foggy", "is_bad_weather",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values(["timestamp", "geohash"]).reset_index(drop=True)


# =============================================================================
# KAYDETME
# =============================================================================

def save(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        print(f"  [UYARI] Boş DataFrame, {path} kaydedilmedi.")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    size_kb = Path(path).stat().st_size // 1024
    print(f"  [Kaydedildi] {path}  ({len(df):,} satır, {size_kb} KB)")

    # Kalite raporu
    kalite = {
        "kaynak":           "Open-Meteo Archive API",
        "nokta_sayisi":     int(df["geohash"].nunique()) if "geohash" in df.columns else 1,
        "toplam_satir":     int(len(df)),
        "eksik_deger":      {col: int(df[col].isna().sum())
                             for col in df.columns if df[col].isna().sum() > 0},
        "tarih_araligi":    {
            "baslangic": str(df["timestamp"].min())[:10] if "timestamp" in df.columns else "",
            "bitis":     str(df["timestamp"].max())[:10] if "timestamp" in df.columns else "",
        },
        "yagisli_saat":     int(df["is_rainy"].sum()) if "is_rainy" in df.columns else 0,
        "karli_saat":       int(df["is_snowy"].sum()) if "is_snowy" in df.columns else 0,
        "firtinali_saat":   int(df["is_stormy"].sum()) if "is_stormy" in df.columns else 0,
        "kotu_hava_saat":   int(df["is_bad_weather"].sum()) if "is_bad_weather" in df.columns else 0,
    }
    kalite_path = str(path).replace(".parquet", "_kalite.json")
    with open(kalite_path, "w", encoding="utf-8") as f:
        json.dump(kalite, f, ensure_ascii=False, indent=2)
    print(f"  [Kalite]     {kalite_path}")


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Open-Meteo Beşiktaş Geçmiş Hava Verisi (geohash bazlı, 58 nokta)"
    )
    parser.add_argument("--start", default="2020-01-01",
                        help="Başlangıç tarihi (varsayılan: 2020-01-01)")
    parser.add_argument("--end",   default="2024-12-31",
                        help="Bitiş tarihi (varsayılan: 2024-12-31)")
    parser.add_argument("--out",   default="gecmis_hava_durumu.parquet")
    args = parser.parse_args()

    df = fetch_historical(args.start, args.end)
    save(df, args.out)

    if not df.empty:
        print(f"\n  Nokta sayısı  : {df['geohash'].nunique()}")
        print(f"  Sıcaklık ort  : {df['temperature_c'].mean():.1f}°C")
        print(f"  Yağışlı satır : {df['is_rainy'].sum():,}")
        print(f"  Karlı satır   : {df['is_snowy'].sum():,}")
        print(f"  Fırtınalı     : {df['is_stormy'].sum():,}")
        print(f"  Zorlu hava    : {df['is_bad_weather'].sum():,} satır")
    print("\n[Tamamlandı]")


if __name__ == "__main__":
    main()
