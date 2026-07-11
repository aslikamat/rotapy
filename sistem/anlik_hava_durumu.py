"""
Open-Meteo Hava Durumu Verisi — Beşiktaş (GEOHASH BAZLI, ÇOKLU NOKTA)
======================================================================
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

────────────────────────────────────────────────────────────────────
BU VERSİYONDAKİ DEĞİŞİKLİK
────────────────────────────────────────────────────────────────────
Önceki sürüm tek bir sabit noktadan (Beşiktaş merkezi) veri çekiyordu.
Bu sürüm, tomtom_besiktas_alansal.py'deki AYNI 58 noktayı kullanarak
hava verisini GEOHASH BAZINDA, konum bazlı çeker. Yani artık her
zaman damgası için 58 satır var (her nokta/geohash için bir tane) —
tıpkı trafik verisindeki gibi.

Open-Meteo API tek istekte birden fazla konumu virgülle ayrılmış
latitude/longitude listesi olarak kabul ediyor ve JSON dizisi (array)
döndürüyor. Bu sayede 58 nokta için 58 ayrı istek yerine YİNE TEK
istek (chunk başına) yapılıyor — istek sayısı artmıyor.

UYARI — veri hacmi: Geçmiş modda (--mod gecmis) 5 yıl × 58 nokta ×
saatlik veri milyonlarca satır üretir (yaklaşık 58 × 5 × 8760 ≈
2.5 milyon satır). Bu normaldir ama diskte ve bellekte yer kaplar;
gerekirse --start/--end ile aralığı daraltın.
"""

import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# BEŞİKTAŞ NOKTALARI
# tomtom_besiktas_alansal.py'deki ROAD_SEGMENTS ile AYNI 58 koordinat
# (yalnızca lat/lon — segment kimliği burada da yok, geohash kullanılıyor)
# =============================================================================

POINTS = [
    {"lat": 41.0428, "lon": 29.0052},
    {"lat": 41.0455, "lon": 29.0058},
    {"lat": 41.048, "lon": 29.0063},
    {"lat": 41.0505, "lon": 29.0068},
    {"lat": 41.053, "lon": 29.0072},
    {"lat": 41.0555, "lon": 29.0075},
    {"lat": 41.0432, "lon": 29.009},
    {"lat": 41.045, "lon": 29.0094},
    {"lat": 41.0467, "lon": 29.0097},
    {"lat": 41.0485, "lon": 29.01},
    {"lat": 41.05, "lon": 29.0103},
    {"lat": 41.0515, "lon": 29.0106},
    {"lat": 41.0395, "lon": 29.006},
    {"lat": 41.04, "lon": 29.0075},
    {"lat": 41.0405, "lon": 29.009},
    {"lat": 41.041, "lon": 29.0105},
    {"lat": 41.0385, "lon": 29.0095},
    {"lat": 41.039, "lon": 29.011},
    {"lat": 41.0395, "lon": 29.008},
    {"lat": 41.0375, "lon": 29.01},
    {"lat": 41.045, "lon": 29.002},
    {"lat": 41.047, "lon": 29.003},
    {"lat": 41.049, "lon": 29.004},
    {"lat": 41.046, "lon": 28.998},
    {"lat": 41.0475, "lon": 28.999},
    {"lat": 41.049, "lon": 29.0},
    {"lat": 41.0415, "lon": 29.004},
    {"lat": 41.043, "lon": 29.0035},
    {"lat": 41.0445, "lon": 29.003},
    {"lat": 41.0675, "lon": 29.011},
    {"lat": 41.0681, "lon": 29.0116},
    {"lat": 41.0665, "lon": 29.01},
    {"lat": 41.0655, "lon": 29.009},
    {"lat": 41.0605, "lon": 29.0075},
    {"lat": 41.062, "lon": 29.008},
    {"lat": 41.06, "lon": 29.01},
    {"lat": 41.044, "lon": 29.0005},
    {"lat": 41.045, "lon": 29.001},
    {"lat": 41.064, "lon": 29.005},
    {"lat": 41.065, "lon": 29.006},
    {"lat": 41.063, "lon": 29.004},
    {"lat": 41.049, "lon": 29.015},
    {"lat": 41.0505, "lon": 29.0155},
    {"lat": 41.052, "lon": 29.016},
    {"lat": 41.0535, "lon": 29.0165},
    {"lat": 41.0479, "lon": 29.018},
    {"lat": 41.0458, "lon": 29.033},
    {"lat": 41.049, "lon": 29.034},
    {"lat": 41.0435, "lon": 29.002},
    {"lat": 41.0415, "lon": 29.001},
    {"lat": 41.058, "lon": 29.013},
    {"lat": 41.056, "lon": 29.012},
    {"lat": 41.0545, "lon": 29.0115},
    {"lat": 41.0422, "lon": 29.0065},
    {"lat": 41.044, "lon": 29.0078},
    {"lat": 41.0408, "lon": 28.9998},
    {"lat": 41.0418, "lon": 29.0028},
    {"lat": 41.0555, "lon": 29.0108},
]

TIMEZONE = "Europe/Istanbul"

# Diğer scriptlerle (trafik, etkinlik) aynı hassasiyet — join için tutarlılık
GEOHASH_PRECISION = 7

# =============================================================================
# GEOHASH — harici kütüphane gerektirmeyen saf Python implementasyonu
# (standart base32 geohash algoritması)
# =============================================================================

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


# Her noktaya kendi geohash'ini bir kez hesaplayıp ekle (tekrar tekrar
# hesaplamamak için önden üretiyoruz)
for _p in POINTS:
    _p["geohash"] = encode_geohash(_p["lat"], _p["lon"], GEOHASH_PRECISION)

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


def _multi_location_params(extra: dict) -> dict:
    """58 noktayı tek istekte sorgulamak için virgülle ayrılmış lat/lon listesi."""
    return {
        "latitude":  ",".join(str(p["lat"]) for p in POINTS),
        "longitude": ",".join(str(p["lon"]) for p in POINTS),
        "hourly":    ",".join(HOURLY_VARS),
        "timezone":  TIMEZONE,
        "wind_speed_unit": "kmh",
        **extra,
    }


def _parse_multi_location_response(raw) -> pd.DataFrame:
    """
    Open-Meteo, birden fazla konum istendiğinde JSON dizisi (array) döner
    (konum sırası isteğe gönderilen sırayla aynıdır). Her elemanı kendi
    POINTS girdisiyle eşleştirip lat/lon/geohash sütunlarını ekleriz.
    """
    entries = raw if isinstance(raw, list) else [raw]
    dfs = []
    for point, entry in zip(POINTS, entries):
        hourly = entry.get("hourly", {})
        if not hourly:
            continue
        df = pd.DataFrame(hourly)
        df["lat"]     = point["lat"]
        df["lon"]     = point["lon"]
        df["geohash"] = point["geohash"]
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def fetch_historical(
    start_date: str = "2020-01-01",
    end_date:   str = "2024-12-31",
) -> pd.DataFrame:
    """
    2020-2024 arası saatlik geçmiş hava verisini, 58 noktanın HER BİRİ
    için (geohash bazlı) çeker. Open-Meteo max ~1 yıllık aralık önerir,
    bu yüzden yıl yıl çekip birleştiriyoruz; her yıl-parçası için TÜM
    58 nokta tek istekte sorgulanır (istek sayısı artmaz).
    """
    print(f"\n[Geçmiş Hava] {start_date} → {end_date} çekiliyor "
          f"({len(POINTS)} nokta/geohash)...")

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

        params = _multi_location_params({
            "start_date": chunk_start,
            "end_date":   chunk_end,
        })

        try:
            r = requests.get(HISTORICAL_URL, params=params, timeout=60)
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
# ANLIK VERİ — api.open-meteo.com/v1/forecast
# =============================================================================

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_anlik(days_ahead: int = 7) -> pd.DataFrame:
    """
    Bugün + days_ahead gün ilerisi için saatlik hava tahmini,
    58 noktanın HER BİRİ için (geohash bazlı). Model inference
    sırasında çağrılır. Tek istekte tüm noktalar sorgulanır.
    """
    print(f"\n[Anlık Hava] Bugün + {days_ahead} gün çekiliyor "
          f"({len(POINTS)} nokta/geohash)...")

    params = _multi_location_params({
        "forecast_days": days_ahead,
        "past_days":     1,   # dünü de ekle (model bağlamı için)
    })

    try:
        r = requests.get(FORECAST_URL, params=params, timeout=30)
        r.raise_for_status()
        df = _parse_multi_location_response(r.json())
        print(f"  {len(df):,} satır ✓")
        return _process(df)
    except Exception as e:
        print(f"  [HATA] {e}")
        return pd.DataFrame()


# =============================================================================
# ORTAK İŞLEME
# =============================================================================

def _process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham API çıktısını LSTM'e hazır formata dönüştürür.
    df zaten lat/lon/geohash sütunlarını içerir (nokta bazında eklendi).
    """
    if df.empty:
        return df

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

    # Zaman özellikleri — date ve hour birleştirildi, minute yok
    # (kaynak veri zaten saatlik olduğu için minute her zaman 0'dır)
    df["timestamp"]  = pd.to_datetime(df["timestamp"])
    df["date_hour"]  = df["timestamp"].dt.strftime("%Y-%m-%d %H")
    df["weekday"]    = df["timestamp"].dt.weekday   # 0=Pzt, 6=Paz
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Hava kodu sınıflandırması
    df["weather_code"] = df["weather_code"].fillna(0).astype(int)
    weather_flags = df["weather_code"].apply(classify_weather).apply(pd.Series)
    df = pd.concat([df, weather_flags], axis=1)

    # Sütun sırası
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


def get_anlik_features(dt: datetime = None, geohash: str = None) -> dict:
    """
    Model inference için tek satır hava feature'ı döner.
    LSTM'e beslenecek anlık hava değişkenlerini verir.

    geohash verilmezse POINTS listesindeki ilk noktanın geohash'i
    kullanılır (Barbaros Bulvarı başlangıcı).

    Kullanım:
        from hava_durumu import get_anlik_features
        features = get_anlik_features(geohash="sxk9kp8")
    """
    df = fetch_anlik(days_ahead=2)
    if df.empty:
        return {}

    target_geohash = geohash or POINTS[0]["geohash"]
    df = df[df["geohash"] == target_geohash]
    if df.empty:
        return {}

    now = dt or datetime.now()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # En yakın saati bul
    df["diff"] = (df["timestamp"] - now).abs()
    row = df.loc[df["diff"].idxmin()]

    return {
        "geohash":          row.get("geohash", target_geohash),
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
        description="Open-Meteo Beşiktaş Hava Verisi (geohash bazlı, çoklu nokta)"
    )
    parser.add_argument("--mod",   choices=["gecmis", "anlik", "her-ikisi"],
                        default="anlik", help="Çalışma modu")
    parser.add_argument("--start", default="2020-01-01", help="Geçmiş mod başlangıç tarihi")
    parser.add_argument("--end",   default="2024-12-31", help="Geçmiş mod bitiş tarihi")
    parser.add_argument("--days",  type=int, default=7, help="Anlık mod: kaç gün ileriye bak")
    parser.add_argument("--out-gecmis", default="besiktas_hava_gecmis.parquet")
    parser.add_argument("--out-anlik",  default="besiktas_hava_anlik.parquet")
    args = parser.parse_args()

    if args.mod in ("gecmis", "her-ikisi"):
        df_g = fetch_historical(args.start, args.end)
        save(df_g, args.out_gecmis)

    if args.mod in ("anlik", "her-ikisi"):
        df_a = fetch_anlik(args.days)
        save(df_a, args.out_anlik)

        if not df_a.empty:
            print(f"\n  Önümüzdeki {args.days} gün ({len(POINTS)} nokta):")
            print(f"  Sıcaklık ort : {df_a['temperature_c'].mean():.1f}°C")
            print(f"  Yağışlı satır: {df_a['is_rainy'].sum()}")
            print(f"  Zorlu hava   : {df_a['is_bad_weather'].sum()} satır")

    print("\n[Tamamlandı]")


if __name__ == "__main__":
    main()