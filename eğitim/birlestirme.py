"""
Birleştirme Scripti — LSTM Eğitim Dataseti
===========================================
Üç veri kaynağını tek tabloda birleştirir:
  1. besiktas_trafik_lstm.parquet      → trafik yoğunluğu (hedef değişken)
  2. besiktas_gecmis_etkinlikler.parquet → etkinlik bilgisi
  3. gecmis_hava_durumu.parquet        → hava durumu

Çalıştırma sırası:
  1. python gecmis_trafik_ibb.py --quick
  2. python gecmis_etkinlik_scraper.py --start 2023-01-01 --end 2024-12-31
  3. python gecmis_hava_durumu.py --start 2023-01-01 --end 2024-12-31
  4. python birlestirme.py

Kurulum:
  pip install pandas pyarrow numpy

Kullanım:
  python birlestirme.py                    # varsayılan
  python birlestirme.py --sample 50000     # 50.000 satır örnekle
  python birlestirme.py --out lstm_data.parquet

Çıktı:
  lstm_egitim_verisi.parquet  → LSTM modeline direkt beslenecek dosya

Sütunlar:
  Hedef  : density (araç yoğunluğu)
  Zaman  : hour, weekday, is_weekend, hour_sin, hour_cos, day_sin, day_cos
  Trafik : speed, lag_1h, lag_24h, lag_7d, ma_3h
  Etkinlik: is_event, event_attendance, hours_to_event
  Hava   : temperature_c, precipitation_mm, wind_speed_kmh,
            is_rainy, is_snowy, is_bad_weather
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# =============================================================================
# DOSYA YOLLARI
# =============================================================================

TRAFIK_FILE   = "besiktas_trafik_lstm.parquet"
ETKINLIK_FILE = "besiktas_gecmis_etkinlikler.parquet"
HAVA_FILE     = "gecmis_hava_durumu.parquet"
OUTPUT_FILE   = "lstm_egitim_verisi.parquet"

# LSTM'e girecek sütunlar (hedef + feature'lar)
LSTM_COLS = [
    # Zaman
    "datetime", "date", "hour", "weekday", "is_weekend",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    # Konum
    "lat", "lon", "location",
    # Trafik (hedef + geçmiş)
    "density", "speed",
    "lag_1h", "lag_24h", "lag_7d", "ma_3h",
    # Etkinlik
    "is_event", "event_attendance", "event_radius_km", "hours_to_event",
    # Hava
    "temperature_c", "precipitation_mm", "wind_speed_kmh",
    "cloud_cover_pct", "humidity_pct",
    "is_rainy", "is_snowy", "is_stormy", "is_bad_weather",
]


# =============================================================================
# 1. TRAFİK VERİSİ YÜKLE
# =============================================================================

def load_trafik(path: str) -> pd.DataFrame:
    print(f"\n[1/3] Trafik verisi yükleniyor: {path}")

    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} bulunamadı.\n"
            f"Önce çalıştırın: python gecmis_trafik_ibb.py --quick"
        )

    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Gereksiz sütunları at
    drop_cols = ["ym", "geohash"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Konum bazında NaN'ları at (lag sütunları)
    # Her lokasyonun ilk 168 saati NaN olabilir — bunları at
    lag_cols = ["lag_1h", "lag_24h", "lag_7d"]
    lag_cols = [c for c in lag_cols if c in df.columns]
    if lag_cols:
        before = len(df)
        df = df.dropna(subset=lag_cols)
        print(f"  lag NaN'ları atıldı: {before - len(df):,} satır")

    print(f"  {len(df):,} satır yüklendi ✓")
    print(f"  Tarih: {df['datetime'].min().date()} → {df['datetime'].max().date()}")
    return df


# =============================================================================
# 2. ETKİNLİK VERİSİ — mesafe bazlı etki yarıçapı ile
# =============================================================================

# Minimum kapasite eşiği — altındaki etkinlikler trafik etkisi yaratmaz
MIN_CAPACITY = 500

# Etki yarıçapı hesabı için taban mesafe (km)
# yarıçap = TABAN_MESAFE × √(kapasite / MIN_CAPACITY)
# Örnekler:
#   500 kişi  → 0.5 km
#   2.350 kişi → 1.1 km
#   5.000 kişi → 1.6 km
#   42.684 kişi → 4.6 km
TABAN_MESAFE_KM = 0.5


def etki_yaricapi(kapasite: float) -> float:
    """Etkinlik kapasitesine göre etki yarıçapı hesaplar (km)."""
    if kapasite < MIN_CAPACITY:
        return 0.0
    return round(TABAN_MESAFE_KM * (kapasite / MIN_CAPACITY) ** 0.5, 2)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    İki koordinat arasındaki mesafeyi km cinsinden hesaplar.
    Haversine formülü — dünya yüzeyindeki kısa mesafeler için doğru.
    """
    R = 6371  # Dünya yarıçapı (km)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def load_etkinlik(path: str, trafik_df: pd.DataFrame) -> pd.DataFrame:
    """
    Etkinlik verisini trafik verisine mesafe bazlı ekler.

    Her trafik satırı için:
      is_event         : lokasyona etkisi olan etkinlik var mı? (0/1)
      event_attendance : varsa en büyük etkinliğin katılımcı sayısı
      event_radius_km  : o etkinliğin etki yarıçapı
      hours_to_event   : en yakın etkinliğe kaç saat kaldı

    Etki yarıçapı = TABAN_MESAFE × √(kapasite / MIN_CAPACITY)
    Lokasyon bu yarıçap içindeyse etkileniyor sayılır.
    """
    print(f"\n[2/3] Etkinlik verisi işleniyor: {path}")

    if not Path(path).exists():
        print(f"  [UYARI] {path} bulunamadı, etkinlik sütunları 0 olarak doldurulacak.")
        trafik_df["is_event"]         = 0
        trafik_df["event_attendance"] = 0
        trafik_df["event_radius_km"]  = 0.0
        trafik_df["hours_to_event"]   = 99.0
        return trafik_df

    ev = pd.read_parquet(path)
    ev["start_datetime"] = pd.to_datetime(ev["start_datetime"], errors="coerce")
    ev["end_datetime"]   = pd.to_datetime(ev["end_datetime"],   errors="coerce")
    ev = ev.dropna(subset=["start_datetime", "end_datetime", "lat", "lon"])

    # Kapasite < 500 olanları filtrele
    ev = ev[ev["estimated_attendance"] >= MIN_CAPACITY].copy()

    # Her etkinliğin etki yarıçapını hesapla
    ev["radius_km"] = ev["estimated_attendance"].apply(etki_yaricapi)

    print(f"  {len(ev)} etkinlik yüklendi (kapasite ≥ {MIN_CAPACITY})")
    print(f"  Yarıçap aralığı: {ev['radius_km'].min():.1f} km "
          f"— {ev['radius_km'].max():.1f} km")

    # Çıktı dizileri
    n                = len(trafik_df)
    is_event         = np.zeros(n, dtype=int)
    event_attendance = np.zeros(n, dtype=float)
    event_radius_km  = np.zeros(n, dtype=float)
    hours_to_event   = np.full(n, 99.0)

    dt_index = pd.DatetimeIndex(trafik_df["datetime"].values)
    lats     = trafik_df["lat"].values
    lons     = trafik_df["lon"].values

    for i, (dt, lat, lon) in enumerate(zip(dt_index, lats, lons)):

        # 1. O an aktif etkinlikler
        active = ev[
            (ev["start_datetime"] <= dt) &
            (ev["end_datetime"]   >= dt)
        ]

        if not active.empty:
            # Her aktif etkinlik için lokasyona mesafeyi hesapla
            for _, row in active.iterrows():
                dist = haversine_km(lat, lon, row["lat"], row["lon"])
                if dist <= row["radius_km"]:
                    # Lokasyon etki yarıçapı içinde
                    if row["estimated_attendance"] > event_attendance[i]:
                        is_event[i]         = 1
                        event_attendance[i] = row["estimated_attendance"]
                        event_radius_km[i]  = row["radius_km"]
                        hours_to_event[i]   = 0.0

        # 2. Etkinlik yoksa — en yakın gelecek etkinliğe kaç saat?
        if is_event[i] == 0:
            future = ev[ev["start_datetime"] > dt]
            if not future.empty:
                # Lokasyona yakın gelecek etkinliklere bak
                for _, row in future.iterrows():
                    dist = haversine_km(lat, lon, row["lat"], row["lon"])
                    if dist <= row["radius_km"]:
                        diff = (row["start_datetime"] - dt).total_seconds() / 3600
                        if diff < hours_to_event[i]:
                            hours_to_event[i] = round(diff, 2)
                        break  # En yakın olanı bulduk

    trafik_df["is_event"]         = is_event
    trafik_df["event_attendance"] = event_attendance
    trafik_df["event_radius_km"]  = event_radius_km
    trafik_df["hours_to_event"]   = hours_to_event.round(2)

    event_count = is_event.sum()
    print(f"  Etkilenen lokasyon-saat: {event_count:,} ({event_count/n*100:.1f}%)")
    print(f"  Örnek yarıçaplar:")
    for cap in [500, 2350, 5000, 42684]:
        print(f"    {cap:>6} kişi → {etki_yaricapi(cap):.1f} km yarıçap")
    return trafik_df


# =============================================================================
# 3. HAVA DURUMU — saate göre birleştir
# =============================================================================

def load_hava(path: str, trafik_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hava verisini trafik verisinin saatiyle eşleştirir.
    Her trafik satırının datetime'ına en yakın hava satırını bulur.
    """
    print(f"\n[3/3] Hava durumu birleştiriliyor: {path}")

    if not Path(path).exists():
        print(f"  [UYARI] {path} bulunamadı, hava sütunları 0 olarak doldurulacak.")
        for col in ["temperature_c", "precipitation_mm", "wind_speed_kmh",
                    "cloud_cover_pct", "humidity_pct",
                    "is_rainy", "is_snowy", "is_stormy", "is_bad_weather"]:
            trafik_df[col] = 0
        return trafik_df

    hava = pd.read_parquet(path)
    hava["timestamp"] = pd.to_datetime(hava["timestamp"])

    # Saate yuvarla — trafik ve hava aynı saate hizalansın
    trafik_df["datetime_saat"] = trafik_df["datetime"].dt.floor("h")
    hava["timestamp_saat"]     = hava["timestamp"].dt.floor("h")

    hava_cols = [
        "timestamp_saat", "temperature_c", "precipitation_mm",
        "wind_speed_kmh", "cloud_cover_pct", "humidity_pct",
        "is_rainy", "is_snowy", "is_stormy", "is_bad_weather",
    ]
    hava_cols = [c for c in hava_cols if c in hava.columns]
    hava_slim = hava[hava_cols].drop_duplicates("timestamp_saat")

    merged = trafik_df.merge(
        hava_slim,
        left_on="datetime_saat",
        right_on="timestamp_saat",
        how="left",
    ).drop(columns=["datetime_saat", "timestamp_saat"], errors="ignore")

    # Hava eşleşmeyen satırlar (tarih aralığı dışı) → 0 doldur
    hava_feature_cols = [
        "temperature_c", "precipitation_mm", "wind_speed_kmh",
        "cloud_cover_pct", "humidity_pct",
        "is_rainy", "is_snowy", "is_stormy", "is_bad_weather",
    ]
    for col in hava_feature_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    eslesen = merged["temperature_c"].ne(0).sum()
    print(f"  {eslesen:,} satır hava verisiyle eşleşti ✓")
    return merged


# =============================================================================
# 4. SON TEMİZLİK ve LSTM HAZIRLIK
# =============================================================================

def finalize(df: pd.DataFrame, sample_n: int = None) -> pd.DataFrame:
    """
    - Gereksiz sütunları at
    - NaN kalan satırları at
    - Örnekle (sample_n verilmişse)
    - Tarihe göre sırala
    """
    print(f"\n[Finalize] Son düzenleme yapılıyor...")
    print(f"  Gelen satır: {len(df):,}")

    # Sadece LSTM sütunlarını al (varsa)
    keep = [c for c in LSTM_COLS if c in df.columns]
    # Listede olmayan ama faydalı olabilecek sütunları da ekle
    extra = [c for c in df.columns if c not in keep and
             c not in ["ym", "geohash", "timestamp_saat", "datetime_saat"]]
    df = df[keep + extra]

    # NaN at
    critical = ["density", "datetime", "lat", "lon"]
    critical = [c for c in critical if c in df.columns]
    before   = len(df)
    df       = df.dropna(subset=critical)
    print(f"  NaN atıldı: {before - len(df):,} satır")

    # Tarihe göre sırala
    df = df.sort_values("datetime").reset_index(drop=True)

    # Örnekleme (isteğe bağlı)
    if sample_n and len(df) > sample_n:
        # Rastgele değil, zaman bazlı örnekle (her N. satırı al)
        step = len(df) // sample_n
        df   = df.iloc[::step].head(sample_n).reset_index(drop=True)
        print(f"  Örnekleme: {sample_n:,} satır seçildi (her {step}. satır)")

    print(f"  Final satır: {len(df):,}")
    print(f"  Final sütun: {len(df.columns)}")
    return df


# =============================================================================
# ÖZET RAPOR
# =============================================================================

def print_summary(df: pd.DataFrame, out_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"  BİRLEŞTİRME TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  Satır sayısı    : {len(df):,}")
    print(f"  Sütun sayısı    : {len(df.columns)}")
    print(f"  Tarih aralığı   : {df['date'].min()} → {df['date'].max()}")

    if "density" in df.columns:
        print(f"  Ort. yoğunluk   : {df['density'].mean():.1f} araç")
        print(f"  Max yoğunluk    : {df['density'].max():.0f} araç")

    if "is_event" in df.columns:
        pct = df["is_event"].mean() * 100
        print(f"  Etkinlik oranı  : %{pct:.1f}")

    if "is_bad_weather" in df.columns:
        pct = df["is_bad_weather"].mean() * 100
        print(f"  Kötü hava oranı : %{pct:.1f}")

    print(f"\n  Sütunlar:")
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_str   = f" ({null_count} NaN)" if null_count > 0 else ""
        print(f"    {col}{null_str}")

    print(f"\n  Çıktı: {out_path}")
    print(f"{'='*60}")


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Beşiktaş LSTM Eğitim Dataseti Birleştirici"
    )
    parser.add_argument("--trafik",   default=TRAFIK_FILE)
    parser.add_argument("--etkinlik", default=ETKINLIK_FILE)
    parser.add_argument("--hava",     default=HAVA_FILE)
    parser.add_argument("--out",      default=OUTPUT_FILE)
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Kaç satır örneklensin (opsiyonel, örn: --sample 100000)"
    )
    args = parser.parse_args()

    # 1. Trafik
    df = load_trafik(args.trafik)

    # 2. Etkinlik
    df = load_etkinlik(args.etkinlik, df)

    # 3. Hava
    df = load_hava(args.hava, df)

    # 4. Finalize
    df = finalize(df, sample_n=args.sample)

    # 5. Kaydet
    df.to_parquet(args.out, index=False, engine="pyarrow")

    # 6. Özet
    print_summary(df, args.out)

    print(f"\n  Sonraki adım: python lstm_egitim.py --data {args.out}")


if __name__ == "__main__":
    main()
