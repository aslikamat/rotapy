"""
Open-Meteo Hava Durumu Verisi — Beşiktaş
==========================================
İki modda çalışır:

  1. GEÇMİŞ (--mod gecmis):
     2020-01-01 → 2024-12-31 arası saatlik hava verisi
     Endpoint: archive-api.open-meteo.com
     LSTM model eğitimi için

  2. ANLIK (--mod anlik):
     Bugün + 7 gün ilerisi
     Endpoint: api.open-meteo.com/v1/forecast
     Model inference için

Kayıt gerekmez, tamamen ücretsiz.

Kurulum:
    pip install requests pandas pyarrow

Kullanım:
    python hava_durumu.py --mod gecmis
    python hava_durumu.py --mod anlik
    python hava_durumu.py --mod gecmis --start 2022-01-01 --end 2023-12-31
    python hava_durumu.py --mod her-ikisi

Çıktı:
    besiktas_hava_gecmis.parquet   → LSTM eğitim verisi
    besiktas_hava_anlik.parquet    → Model inference verisi

Sütunlar:
    timestamp, date, hour, weekday, is_weekend
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
"""

import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# BEŞİKTAŞ KOORDİNATLARI
# =============================================================================

LAT = 41.0402
LON = 29.0097
TIMEZONE = "Europe/Istanbul"

# =============================================================================
# WMO HAVA KODU SINIFLANDIRMASI
# Kaynak: https://open-meteo.com/en/docs — WMO Weather interpretation codes
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
        "is_rainy":      int(51 <= code <= 67 or 80 <= code <= 82),
        "is_snowy":      int(71 <= code <= 77 or 85 <= code <= 86),
        "is_stormy":     int(95 <= code <= 99),
        "is_foggy":      int(45 <= code <= 48),
        "is_bad_weather": int(code >= 45),   # ana LSTM feature
    }

# =============================================================================
# GEÇMİŞ VERİ — archive-api.open-meteo.com
# =============================================================================

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "relative_humidity_2m",
    "weather_code",
    "snow_depth",
]

def fetch_historical(
    start_date: str = "2020-01-01",
    end_date:   str = "2024-12-31",
) -> pd.DataFrame:
    """
    2020-2024 arası saatlik geçmiş hava verisini çeker.
    Open-Meteo max ~1 yıllık aralık önerir,
    bu yüzden yıl yıl çekip birleştiriyoruz.
    """
    print(f"\n[Geçmiş Hava] {start_date} → {end_date} çekiliyor...")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    # Yıllık parçalara böl (API limiti için)
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(
            datetime(current.year, 12, 31),
            end
        )
        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))
        current = datetime(current.year + 1, 1, 1)

    all_dfs = []
    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] {chunk_start} → {chunk_end}...", end=" ")

        params = {
            "latitude":   LAT,
            "longitude":  LON,
            "start_date": chunk_start,
            "end_date":   chunk_end,
            "hourly":     ",".join(HOURLY_VARS),
            "timezone":   TIMEZONE,
            "wind_speed_unit": "kmh",
        }

        try:
            r = requests.get(HISTORICAL_URL, params=params, timeout=30)
            r.raise_for_status()
            hourly = r.json()["hourly"]
            df     = pd.DataFrame(hourly)
            all_dfs.append(df)
            print(f"{len(df):,} satır ✓")
        except Exception as e:
            print(f"[HATA] {e}")

    if not all_dfs:
        return pd.DataFrame()

    merged = pd.concat(all_dfs, ignore_index=True)
    return _process(merged)


# =============================================================================
# ANLIK VERİ — api.open-meteo.com/v1/forecast
# =============================================================================

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_anlik(days_ahead: int = 7) -> pd.DataFrame:
    """
    Bugün + days_ahead gün ilerisi için saatlik hava tahmini.
    Model inference sırasında çağrılır.
    """
    print(f"\n[Anlık Hava] Bugün + {days_ahead} gün çekiliyor...")

    params = {
        "latitude":       LAT,
        "longitude":      LON,
        "hourly":         ",".join(HOURLY_VARS),
        "timezone":       TIMEZONE,
        "wind_speed_unit": "kmh",
        "forecast_days":  days_ahead,
        "past_days":      1,   # dünü de ekle (model bağlamı için)
    }

    try:
        r = requests.get(FORECAST_URL, params=params, timeout=15)
        r.raise_for_status()
        hourly = r.json()["hourly"]
        df     = pd.DataFrame(hourly)
        print(f"  {len(df):,} satır ✓")
        return _process(df)
    except Exception as e:
        print(f"  [HATA] {e}")
        return pd.DataFrame()


# =============================================================================
# ORTAK İŞLEME
# =============================================================================

def _process(df: pd.DataFrame) -> pd.DataFrame:
    """Ham API çıktısını LSTM'e hazır formata dönüştürür."""

    # Sütun yeniden adlandırma
    df = df.rename(columns={
        "time":                   "timestamp",
        "temperature_2m":         "temperature_c",
        "precipitation":          "precipitation_mm",
        "wind_speed_10m":         "wind_speed_kmh",
        "cloud_cover":            "cloud_cover_pct",
        "relative_humidity_2m":   "humidity_pct",
        "weather_code":           "weather_code",
        "snow_depth":             "snow_depth_m",
    })

    # Zaman özellikleri
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date.astype(str)
    df["hour"]      = df["timestamp"].dt.hour
    df["weekday"]   = df["timestamp"].dt.weekday   # 0=Pzt, 6=Paz
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Hava kodu sınıflandırması
    df["weather_code"] = df["weather_code"].fillna(0).astype(int)
    weather_flags = df["weather_code"].apply(classify_weather).apply(pd.Series)
    df = pd.concat([df, weather_flags], axis=1)

    # Koordinat ekle (birleştirme scriptinde join için)
    df["lat"] = LAT
    df["lon"] = LON

    # Sütun sırası
    cols = [
        "timestamp", "date", "hour", "weekday", "is_weekend",
        "lat", "lon",
        "temperature_c", "precipitation_mm", "wind_speed_kmh",
        "cloud_cover_pct", "humidity_pct", "snow_depth_m",
        "weather_code",
        "is_rainy", "is_snowy", "is_stormy", "is_foggy", "is_bad_weather",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].reset_index(drop=True)


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


def get_anlik_features(dt: datetime = None) -> dict:
    """
    Model inference için tek satır hava feature'ı döner.
    LSTM'e beslenecek anlık hava değişkenlerini verir.

    Kullanım:
        from hava_durumu import get_anlik_features
        features = get_anlik_features()
    """
    df = fetch_anlik(days_ahead=2)
    if df.empty:
        return {}

    now = dt or datetime.now()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # En yakın saati bul
    df["diff"] = (df["timestamp"] - now).abs()
    row = df.loc[df["diff"].idxmin()]

    return {
        "temperature_c":    row.get("temperature_c",    0),
        "precipitation_mm": row.get("precipitation_mm", 0),
        "wind_speed_kmh":   row.get("wind_speed_kmh",   0),
        "cloud_cover_pct":  row.get("cloud_cover_pct",  0),
        "humidity_pct":     row.get("humidity_pct",     0),
        "is_rainy":         int(row.get("is_rainy",     0)),
        "is_snowy":         int(row.get("is_snowy",     0)),
        "is_stormy":        int(row.get("is_stormy",    0)),
        "is_bad_weather":   int(row.get("is_bad_weather", 0)),
    }


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Open-Meteo Beşiktaş Geçmiş Hava Verisi (2020-2024)"
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--out",   default="gecmis_hava_durumu.parquet")
    args = parser.parse_args()

    df = fetch_historical(args.start, args.end)
    save(df, args.out)

    if not df.empty:
        print(f"\n  Sıcaklık ort : {df['temperature_c'].mean():.1f}°C")
        print(f"  Yağışlı saat : {df['is_rainy'].sum():,}")
        print(f"  Karlı saat   : {df['is_snowy'].sum():,}")
        print(f"  Fırtınalı    : {df['is_stormy'].sum():,}")
        print(f"  Zorlu hava   : {df['is_bad_weather'].sum():,} saat")
    print("\n[Tamamlandı]")


if __name__ == "__main__":
    main()
