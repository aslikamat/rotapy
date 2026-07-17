"""
Birleştirme Scripti — LSTM Eğitim Dataseti
===========================================
Üç veri kaynağını tek tabloda birleştirir:
  1. besiktas_trafik_lstm.parquet        → trafik yoğunluğu (hedef değişken)
  2. besiktas_gecmis_etkinlikler.parquet → etkinlik bilgisi
  3. gecmis_hava_durumu.parquet          → hava durumu

Birleştirme anahtarı: geohash + date_hour
  - geohash   : lat/lon'dan üretilen 7 haneli konum kodu
  - date_hour : "YYYY-MM-DD HH" formatı (dakika yok)

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
  Anahtar : geohash, date_hour
  Zaman   : weekday, is_weekend, hour_sin, hour_cos, day_sin, day_cos
  Konum   : lat, lon
  Trafik  : density, speed, lag_1h, lag_24h, lag_7d, ma_3h
  Etkinlik: is_event, event_attendance, event_radius_km, hours_to_event
  Hava    : temperature_c, precipitation_mm, wind_speed_kmh,
            cloud_cover_pct, humidity_pct,
            is_rainy, is_snowy, is_stormy, is_bad_weather
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

# LSTM'e girecek sütunlar
LSTM_COLS = [
    # Anahtar / kimlik
    "datetime", "date_hour", "weekday", "is_weekend",
    # Döngüsel zaman
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    # Konum
    "lat", "lon", "geohash",
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
# GEOHASH — anlik scriptlerle AYNI implementasyon
# =============================================================================

_GEOHASH_BASE32   = "0123456789bcdefghjkmnpqrstuvwxyz"
GEOHASH_PRECISION = 7


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

    # date_hour yoksa datetime'dan üret
    if "date_hour" not in df.columns:
        df["date_hour"] = df["datetime"].dt.strftime("%Y-%m-%d %H")

    # geohash yoksa lat/lon'dan üret
    if "geohash" not in df.columns and "lat" in df.columns and "lon" in df.columns:
        print("  geohash üretiliyor...")
        df["geohash"] = df.apply(
            lambda r: encode_geohash(r["lat"], r["lon"])
            if pd.notna(r["lat"]) and pd.notna(r["lon"]) else "",
            axis=1
        )

    # Gereksiz sütunları at
    drop_cols = ["ym", "location", "date", "hour"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # lag NaN'larını at
    lag_cols = [c for c in ["lag_1h", "lag_24h", "lag_7d"] if c in df.columns]
    if lag_cols:
        before = len(df)
        df = df.dropna(subset=lag_cols)
        print(f"  lag NaN'ları atıldı: {before - len(df):,} satır")

    print(f"  {len(df):,} satır yüklendi ✓")
    print(f"  Tarih: {df['date_hour'].min()[:10]} → {df['date_hour'].max()[:10]}")
    return df


# =============================================================================
# 2. ETKİNLİK VERİSİ — mesafe bazlı etki yarıçapı
# =============================================================================

MIN_CAPACITY    = 500
TABAN_MESAFE_KM = 0.5


def etki_yaricapi(kapasite: float) -> float:
    if kapasite < MIN_CAPACITY:
        return 0.0
    return round(TABAN_MESAFE_KM * (kapasite / MIN_CAPACITY) ** 0.5, 2)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R    = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a    = (np.sin(dlat / 2) ** 2 +
            np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
            np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def load_etkinlik(path: str, trafik_df: pd.DataFrame) -> pd.DataFrame:
    """
    Etkinlik verisini trafik verisine mesafe bazlı ekler.
    Birleştirme anahtarı: start_datetime / end_datetime + koordinat mesafesi.

    Her trafik satırı için:
      is_event         : etki yarıçapı içinde etkinlik var mı? (0/1)
      event_attendance : varsa en büyük etkinliğin katılımcı sayısı
      event_radius_km  : etki yarıçapı (km)
      hours_to_event   : en yakın etkinliğe kaç saat kaldı
    """
    print(f"\n[2/3] Etkinlik verisi işleniyor: {path}")

    if not Path(path).exists():
        print(f"  [UYARI] {path} bulunamadı, etkinlik sütunları 0 doldurulacak.")
        trafik_df["is_event"]         = 0
        trafik_df["event_attendance"] = 0
        trafik_df["event_radius_km"]  = 0.0
        trafik_df["hours_to_event"]   = 99.0
        return trafik_df

    ev = pd.read_parquet(path)

    # start_datetime / end_datetime — her iki formatta da dene
    for col in ["start_datetime", "end_datetime"]:
        if col in ev.columns:
            ev[col] = pd.to_datetime(ev[col], errors="coerce")

    # start_date_hour'dan start_datetime üret (eski format fallback)
    if "start_datetime" not in ev.columns and "start_date_hour" in ev.columns:
        ev["start_datetime"] = pd.to_datetime(
            ev["start_date_hour"], format="%Y-%m-%d %H", errors="coerce"
        )
    if "end_datetime" not in ev.columns and "end_date_hour" in ev.columns:
        ev["end_datetime"] = pd.to_datetime(
            ev["end_date_hour"], format="%Y-%m-%d %H", errors="coerce"
        )

    ev = ev.dropna(subset=["start_datetime", "end_datetime", "lat", "lon"])
    ev = ev[ev["estimated_attendance"] >= MIN_CAPACITY].copy()
    ev["radius_km"] = ev["estimated_attendance"].apply(etki_yaricapi)

    print(f"  {len(ev)} etkinlik yüklendi (kapasite ≥ {MIN_CAPACITY})")
    if not ev.empty:
        print(f"  Yarıçap aralığı: {ev['radius_km'].min():.1f} km "
              f"— {ev['radius_km'].max():.1f} km")

    n                = len(trafik_df)
    is_event         = np.zeros(n, dtype=int)
    event_attendance = np.zeros(n, dtype=float)
    event_radius_km  = np.zeros(n, dtype=float)
    hours_to_event   = np.full(n, 99.0)

    dt_index = pd.DatetimeIndex(trafik_df["datetime"].values)
    lats     = trafik_df["lat"].values
    lons     = trafik_df["lon"].values

    for i, (dt, lat, lon) in enumerate(zip(dt_index, lats, lons)):
        # Aktif etkinlikler
        active = ev[
            (ev["start_datetime"] <= dt) &
            (ev["end_datetime"]   >= dt)
        ]
        if not active.empty:
            for _, row in active.iterrows():
                dist = haversine_km(lat, lon, row["lat"], row["lon"])
                if dist <= row["radius_km"]:
                    if row["estimated_attendance"] > event_attendance[i]:
                        is_event[i]         = 1
                        event_attendance[i] = row["estimated_attendance"]
                        event_radius_km[i]  = row["radius_km"]
                        hours_to_event[i]   = 0.0

        # En yakın gelecek etkinlik
        if is_event[i] == 0:
            future = ev[ev["start_datetime"] > dt]
            for _, row in future.iterrows():
                dist = haversine_km(lat, lon, row["lat"], row["lon"])
                if dist <= row["radius_km"]:
                    diff = (row["start_datetime"] - dt).total_seconds() / 3600
                    if diff < hours_to_event[i]:
                        hours_to_event[i] = round(diff, 2)
                    break

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
# 3. HAVA DURUMU — geohash + date_hour ile birleştir
# =============================================================================

def load_hava(path: str, trafik_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hava verisini geohash + date_hour anahtarıyla trafik verisine join eder.
    Aynı geohash bölgesindeki aynı saatin hava koşulunu getirir.
    geohash eşleşmezse sadece date_hour ile tekrar dener (alan ortalaması).
    """
    print(f"\n[3/3] Hava durumu birleştiriliyor: {path}")

    hava_feature_cols = [
        "temperature_c", "precipitation_mm", "wind_speed_kmh",
        "cloud_cover_pct", "humidity_pct",
        "is_rainy", "is_snowy", "is_stormy", "is_bad_weather",
    ]

    if not Path(path).exists():
        print(f"  [UYARI] {path} bulunamadı, hava sütunları 0 doldurulacak.")
        for col in hava_feature_cols:
            trafik_df[col] = 0
        return trafik_df

    hava = pd.read_parquet(path)

    # date_hour sütununu tespit et / üret
    if "date_hour" not in hava.columns:
        if "timestamp" in hava.columns:
            hava["timestamp"] = pd.to_datetime(hava["timestamp"])
            hava["date_hour"] = hava["timestamp"].dt.strftime("%Y-%m-%d %H")
        else:
            print("  [UYARI] Hava verisinde date_hour bulunamadı.")
            for col in hava_feature_cols:
                trafik_df[col] = 0
            return trafik_df

    keep = ["date_hour", "geohash"] + [c for c in hava_feature_cols if c in hava.columns]
    hava_slim = hava[[c for c in keep if c in hava.columns]].copy()

    # ── Önce geohash + date_hour ile dene ──────────────────────────────────
    if "geohash" in hava_slim.columns and "geohash" in trafik_df.columns:
        hava_by_geo = (
            hava_slim
            .groupby(["geohash", "date_hour"])
            [hava_feature_cols]
            .mean()
            .reset_index()
        )
        merged = trafik_df.merge(
            hava_by_geo,
            on=["geohash", "date_hour"],
            how="left",
        )
        eslesen = merged["temperature_c"].notna().sum()
        print(f"  geohash + date_hour ile {eslesen:,} satır eşleşti")
    else:
        merged = trafik_df.copy()
        eslesen = 0

    # ── Eşleşmeyenler için sadece date_hour ile doldur ─────────────────────
    eksik = merged["temperature_c"].isna().sum() if "temperature_c" in merged.columns else len(merged)
    if eksik > 0:
        hava_by_hour = (
            hava_slim
            .groupby("date_hour")
            [hava_feature_cols]
            .mean()
            .reset_index()
        )
        # Suffix ile çakışmayı önle
        merged2 = merged.merge(
            hava_by_hour,
            on="date_hour",
            how="left",
            suffixes=("", "_fallback"),
        )
        for col in hava_feature_cols:
            fb = col + "_fallback"
            if fb in merged2.columns:
                merged2[col] = merged2[col].fillna(merged2[fb])
                merged2 = merged2.drop(columns=[fb])
        merged = merged2
        print(f"  date_hour fallback ile {eksik:,} eksik satır dolduruldu")

    # Hâlâ eksik kalanları 0 ile doldur
    for col in hava_feature_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    toplam_eslesen = merged["temperature_c"].ne(0).sum()
    print(f"  Toplam eşleşen: {toplam_eslesen:,} / {len(merged):,} satır ✓")
    return merged


# =============================================================================
# 4. SON TEMİZLİK ve LSTM HAZIRLIK
# =============================================================================

def finalize(df: pd.DataFrame, sample_n: int = None) -> pd.DataFrame:
    print(f"\n[Finalize] Son düzenleme yapılıyor...")
    print(f"  Gelen satır: {len(df):,}")

    # LSTM sütunları + ek faydalı sütunlar
    keep  = [c for c in LSTM_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in keep and
             c not in ["ym", "location", "date", "hour",
                       "timestamp_saat", "datetime_saat"]]
    df = df[keep + extra]

    # Kritik NaN'ları at
    critical = [c for c in ["density", "datetime", "lat", "lon"] if c in df.columns]
    before   = len(df)
    df       = df.dropna(subset=critical)
    print(f"  NaN atıldı: {before - len(df):,} satır")

    df = df.sort_values("datetime").reset_index(drop=True)

    if sample_n and len(df) > sample_n:
        step = len(df) // sample_n
        df   = df.iloc[::step].head(sample_n).reset_index(drop=True)
        print(f"  Örnekleme: {sample_n:,} satır (her {step}. satır)")

    print(f"  Final satır : {len(df):,}")
    print(f"  Final sütun : {len(df.columns)}")
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

    date_col = "date_hour" if "date_hour" in df.columns else "datetime"
    print(f"  Tarih aralığı   : {str(df[date_col].min())[:10]} "
          f"→ {str(df[date_col].max())[:10]}")

    if "density" in df.columns:
        print(f"  Ort. yoğunluk   : {df['density'].mean():.1f} araç")
        print(f"  Max yoğunluk    : {df['density'].max():.0f} araç")

    if "geohash" in df.columns:
        print(f"  Benzersiz geohash: {df['geohash'].nunique()}")

    if "is_event" in df.columns:
        print(f"  Etkinlik oranı  : %{df['is_event'].mean()*100:.1f}")

    if "is_bad_weather" in df.columns:
        print(f"  Kötü hava oranı : %{df['is_bad_weather'].mean()*100:.1f}")

    print(f"\n  Sütunlar:")
    for col in df.columns:
        null_n   = df[col].isna().sum()
        null_str = f" ({null_n} NaN)" if null_n > 0 else ""
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
    parser.add_argument("--sample",   type=int, default=None,
                        help="Kaç satır örneklensin (örn: --sample 100000)")
    args = parser.parse_args()

    df = load_trafik(args.trafik)
    df = load_etkinlik(args.etkinlik, df)
    df = load_hava(args.hava, df)
    df = finalize(df, sample_n=args.sample)

    df.to_parquet(args.out, index=False, engine="pyarrow")
    print_summary(df, args.out)

    print(f"\n  Sonraki adım: python lstm_egitim.py --data {args.out}")


if __name__ == "__main__":
    main()
