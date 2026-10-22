"""
Birleştirme Scripti — LSTM Eğitim Dataseti
===========================================
DB şemasıyla uyumlu sütun yapısı:
  LSTM_DESTEKLİ_İBB_BEŞİKTAŞ_VERİSİ tablosu → trafik verisi
  HavaDurumuVerisi tablosu                    → hava verisi
  EtkinlikVerisi tablosu                      → etkinlik verisi

Birleştirme anahtarı: geohash + date_hour

Çalıştırma sırası:
  1. python gecmis_trafik_ibb.py --quick
  2. python gecmis_etkinlik_scraper.py --start 2023-01-01 --end 2024-12-31
  3. python gecmis_hava_durumu.py --start 2023-01-01 --end 2024-12-31
  4. python birlestirme.py

Kurulum:
  pip install pandas pyarrow numpy

Kullanım:
  python birlestirme.py
  python birlestirme.py --sample 50000
  python birlestirme.py --out lstm_data.parquet

Çıktı:
  lstm_egitim_verisi.parquet → DB şemasıyla uyumlu LSTM eğitim dosyası

DB Sütun Eşleştirmesi:
  LSTM_DESTEKLİ_İBB_BEŞİKTAŞ_VERİSİ:
    timestamp, date_hour, weekday, is_weekend, is_event_time
    lat, lon, geohash, length_m
    current_speed, freeflow_speed, congestion_ratio
    delay_ratio, travel_time_s, freeflow_time_s
    confidence, road_closure

  HavaDurumuVerisi:
    timestamp, date_hour, weekday, is_weekend
    lat, lon, geohash
    temperature_c, precipitation_mm, wind_speed_kmh
    cloud_cover_pct, humidity_pct, snow_depth_m
    weather_code, is_rainy, is_snowy, is_stormy, is_foggy, is_bad_weather

  EtkinlikVerisi:
    id, source, name, category
    start_date_hour, end_date_hour
    start_datetime, end_datetime
    venue, lat, lon, geohash
    estimated_attendance, is_active, url
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime

# =============================================================================
# DOSYA YOLLARI
# =============================================================================

TRAFIK_FILE   = "besiktas_trafik_lstm.parquet"
ETKINLIK_FILE = "besiktas_gecmis_etkinlikler.parquet"
HAVA_FILE     = "gecmis_hava_durumu.parquet"
OUTPUT_FILE   = "lstm_egitim_verisi.parquet"

# =============================================================================
# DB ŞEMASINA UYUMLU LSTM SÜTUNLARI
# =============================================================================

# LSTM_DESTEKLİ_İBB_BEŞİKTAŞ_VERİSİ sütunları
TRAFIK_COLS = [
    "timestamp", "date_hour", "weekday", "is_weekend",
    "is_event_time",                          # etkinlik flag'i (bool)
    "lat", "lon", "geohash", "length_m",
    "current_speed", "freeflow_speed",
    "congestion_ratio",                        # hedef değişken
    "delay_ratio", "travel_time_s", "freeflow_time_s",
    "confidence", "road_closure",
]

# HavaDurumuVerisi sütunları
HAVA_COLS = [
    "temperature_c", "precipitation_mm", "wind_speed_kmh",
    "cloud_cover_pct", "humidity_pct", "snow_depth_m",
    "weather_code",
    "is_rainy", "is_snowy", "is_stormy", "is_foggy", "is_bad_weather",
]

# EtkinlikVerisi sütunları (birleştirmede kullanılan)
ETKINLIK_JOIN_COLS = [
    "start_date_hour", "end_date_hour",
    "start_datetime", "end_datetime",
    "lat", "lon", "geohash",
    "estimated_attendance", "name", "category", "venue",
]

# LSTM'e girecek tüm feature sütunları
LSTM_COLS = (
    TRAFIK_COLS +
    HAVA_COLS +
    [
        # Etkinlik etkisi (load_etkinlik'ten üretilir)
        "is_event", "event_attendance", "event_radius_km", "hours_to_event",
        # Döngüsel zaman (IBB verisinde yoksa üretilir)
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        # Lag feature'ları
        "lag_1h", "lag_24h", "lag_7d", "ma_3h",
    ]
)

# =============================================================================
# GEOHASH — tüm scriptlerle AYNI implementasyon
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
# 1. TRAFİK VERİSİ YÜKLE — DB şemasına dönüştür
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

    # ── timestamp → DB şemasıyla uyumlu ──────────────────────────────────
    if "timestamp" not in df.columns:
        df["timestamp"] = df["datetime"]

    # ── date_hour ─────────────────────────────────────────────────────────
    if "date_hour" not in df.columns:
        df["date_hour"] = df["datetime"].dt.strftime("%Y-%m-%d %H")

    # ── geohash ───────────────────────────────────────────────────────────
    if "geohash" not in df.columns and "lat" in df.columns:
        print("  geohash üretiliyor...")
        df["geohash"] = df.apply(
            lambda r: encode_geohash(r["lat"], r["lon"])
            if pd.notna(r["lat"]) and pd.notna(r["lon"]) else "",
            axis=1
        )

    # ── IBB sütunlarını DB şemasına eşleştir ──────────────────────────────
    # IBB: density, speed → DB: congestion_ratio, current_speed
    col_map = {
        "speed":           "current_speed",
        "minimum_speed":   "freeflow_speed",   # IBB minimum_speed ≈ serbest akış
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # congestion_ratio — DB'nin hedef değişkeni
    # IBB'de congestion_ratio yoksa density / max_density ile hesapla
    if "congestion_ratio" not in df.columns:
        if "density" in df.columns and "maximum_speed" in df.columns:
            # Araç yoğunluğunu normalize et (0=boş, 1=tam dolu)
            max_d = df["density"].quantile(0.95)
            df["congestion_ratio"] = (df["density"] / max_d).clip(0, 1).round(4)
        elif "current_speed" in df.columns and "freeflow_speed" in df.columns:
            df["congestion_ratio"] = (
                1 - df["current_speed"] / df["freeflow_speed"].replace(0, np.nan)
            ).clip(0, 1).fillna(0).round(4)
        else:
            df["congestion_ratio"] = 0.0

    # delay_ratio
    if "delay_ratio" not in df.columns:
        if "travel_time_s" in df.columns and "freeflow_time_s" in df.columns:
            df["delay_ratio"] = (
                (df["travel_time_s"] - df["freeflow_time_s"]) /
                df["freeflow_time_s"].replace(0, np.nan)
            ).clip(0, None).fillna(0).round(4)
        else:
            df["delay_ratio"] = df["congestion_ratio"]

    # DB'de olmayan sütunları varsayılan değerlerle doldur
    defaults = {
        "length_m":        300,    # varsayılan segment uzunluğu
        "travel_time_s":   0,
        "freeflow_time_s": 0,
        "confidence":      1.0,
        "road_closure":    False,
        "is_event_time":   False,  # load_etkinlik'te güncellenecek
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    # Döngüsel zaman feature'ları (DB şemasında yok ama LSTM için gerekli)
    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
    df["day_sin"]  = np.sin(2 * np.pi * df["datetime"].dt.weekday / 7)
    df["day_cos"]  = np.cos(2 * np.pi * df["datetime"].dt.weekday / 7)

    # Gereksiz sütunları at
    drop_cols = ["ym", "location", "date", "hour"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # lag NaN'larını at
    lag_cols = [c for c in ["lag_1h", "lag_24h", "lag_7d"] if c in df.columns]
    lag_nan_silinen = 0
    if lag_cols:
        before = len(df)
        df = df.dropna(subset=lag_cols)
        lag_nan_silinen = before - len(df)
        print(f"  lag NaN'ları atıldı: {lag_nan_silinen:,} satır")
    df.attrs["lag_nan_silinen"] = lag_nan_silinen

    print(f"  {len(df):,} satır yüklendi ✓")
    print(f"  Tarih: {df['date_hour'].min()[:10]} → {df['date_hour'].max()[:10]}")
    if "congestion_ratio" in df.columns:
        print(f"  Ort. congestion_ratio: {df['congestion_ratio'].mean():.3f}")
    return df


# =============================================================================
# 2. ETKİNLİK VERİSİ — EtkinlikVerisi şemasına uyumlu
# =============================================================================

MIN_CAPACITY    = 500
TABAN_MESAFE_KM = 0.5


def etki_yaricapi(kapasite: float) -> float:
    if kapasite < MIN_CAPACITY:
        return 0.0
    return round(TABAN_MESAFE_KM * (kapasite / MIN_CAPACITY) ** 0.5, 2)


def haversine_km_vektor(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    R    = 6371.0
    lat1 = np.radians(lat1[:, None])
    lon1 = np.radians(lon1[:, None])
    lat2 = np.radians(lat2[None, :])
    lon2 = np.radians(lon2[None, :])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_etkinlik(path: str, trafik_df: pd.DataFrame) -> pd.DataFrame:
    """
    EtkinlikVerisi tablosundan veriyi okur, trafik verisine mesafe bazlı ekler.

    DB şemasıyla uyumlu sütunlar:
      start_date_hour, end_date_hour → aktif etkinlik kontrolü
      start_datetime, end_datetime   → hassas zaman kontrolü
      lat, lon, geohash              → mesafe hesabı
      estimated_attendance           → etki yarıçapı hesabı
      is_active                      → şu an aktif mi?

    Trafik verisine eklenen sütunlar:
      is_event_time    : DB şemasındaki boolean (LSTM_DESTEKLİ tablosunda)
      is_event         : etki yarıçapı içinde etkinlik var mı?
      event_attendance : en büyük etkinliğin katılımcı sayısı
      event_radius_km  : etki yarıçapı (km)
      hours_to_event   : en yakın gelecek etkinliğe kaç saat kaldı
    """
    print(f"\n[2/3] Etkinlik verisi işleniyor: {path}")

    if not Path(path).exists():
        print(f"  [UYARI] {path} bulunamadı, etkinlik sütunları 0 doldurulacak.")
        trafik_df["is_event_time"]    = False
        trafik_df["is_event"]         = 0
        trafik_df["event_attendance"] = 0
        trafik_df["event_radius_km"]  = 0.0
        trafik_df["hours_to_event"]   = 99.0
        return trafik_df

    ev = pd.read_parquet(path)

    # start_datetime / end_datetime — DB şemasındaki timestamp sütunları
    for col in ["start_datetime", "end_datetime"]:
        if col in ev.columns:
            ev[col] = pd.to_datetime(ev[col], errors="coerce")

    # start_date_hour fallback
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
    ev = ev.reset_index(drop=True)

    print(f"  {len(ev)} etkinlik yüklendi (kapasite ≥ {MIN_CAPACITY})")
    if not ev.empty:
        print(f"  Yarıçap: {ev['radius_km'].min():.1f} km — {ev['radius_km'].max():.1f} km")

    n                = len(trafik_df)
    is_event         = np.zeros(n, dtype=int)
    event_attendance = np.zeros(n, dtype=float)
    event_radius_km  = np.zeros(n, dtype=float)
    hours_to_event   = np.full(n, 99.0)

    t_lats = trafik_df["lat"].values.astype(float)
    t_lons = trafik_df["lon"].values.astype(float)
    t_dts  = trafik_df["datetime"].values

    ev_starts = ev["start_datetime"].values
    ev_ends   = ev["end_datetime"].values
    ev_lats   = ev["lat"].values.astype(float)
    ev_lons   = ev["lon"].values.astype(float)
    ev_caps   = ev["estimated_attendance"].values.astype(float)
    ev_radii  = ev["radius_km"].values.astype(float)

    unique_hours = np.unique(t_dts.astype("datetime64[h]"))
    print(f"  {len(unique_hours)} benzersiz saat işleniyor (vektörize)...")

    for hour in unique_hours:
        hour_pd  = pd.Timestamp(hour)
        mask_t   = (trafik_df["datetime"].dt.floor("h") == hour_pd).values
        mask_ev  = (ev_starts <= hour_pd) & (ev_ends >= hour_pd)

        if mask_t.any() and mask_ev.any():
            dist_matrix = haversine_km_vektor(
                t_lats[mask_t], t_lons[mask_t],
                ev_lats[mask_ev], ev_lons[mask_ev]
            )
            etki_matrix = dist_matrix <= ev_radii[mask_ev][None, :]
            t_indices   = np.where(mask_t)[0]

            for ti, row_etki in enumerate(etki_matrix):
                if row_etki.any():
                    etkilenen_caps = ev_caps[mask_ev][row_etki]
                    en_buyuk       = np.argmax(etkilenen_caps)
                    ev_idx         = np.where(mask_ev)[0][np.where(row_etki)[0][en_buyuk]]
                    is_event[t_indices[ti]]         = 1
                    event_attendance[t_indices[ti]] = ev_caps[ev_idx]
                    event_radius_km[t_indices[ti]]  = ev_radii[ev_idx]
                    hours_to_event[t_indices[ti]]   = 0.0

        no_event_mask = mask_t & (is_event == 0)
        if no_event_mask.any():
            future_mask = ev_starts > hour_pd
            if future_mask.any():
                dist_matrix = haversine_km_vektor(
                    t_lats[no_event_mask], t_lons[no_event_mask],
                    ev_lats[future_mask],  ev_lons[future_mask]
                )
                etki_matrix = dist_matrix <= ev_radii[future_mask][None, :]
                t_indices   = np.where(no_event_mask)[0]
                for ti, row_etki in enumerate(etki_matrix):
                    if row_etki.any():
                        yakin_idx  = np.where(row_etki)[0][0]
                        diff_hours = (
                            pd.Timestamp(ev_starts[future_mask][yakin_idx]) - hour_pd
                        ).total_seconds() / 3600
                        if diff_hours < hours_to_event[t_indices[ti]]:
                            hours_to_event[t_indices[ti]] = round(diff_hours, 2)

    trafik_df["is_event"]         = is_event
    trafik_df["event_attendance"] = event_attendance
    trafik_df["event_radius_km"]  = event_radius_km
    trafik_df["hours_to_event"]   = hours_to_event.round(2)

    # DB şemasındaki is_event_time (boolean) — is_event'ten türetilir
    trafik_df["is_event_time"] = trafik_df["is_event"].astype(bool)

    event_count = is_event.sum()
    print(f"  Etkilenen lokasyon-saat: {event_count:,} ({event_count/n*100:.1f}%)")
    return trafik_df


# =============================================================================
# 3. HAVA DURUMU — HavaDurumuVerisi şemasına uyumlu
# =============================================================================

def load_hava(path: str, trafik_df: pd.DataFrame) -> pd.DataFrame:
    """
    HavaDurumuVerisi tablosundan okur, geohash + date_hour ile join eder.
    DB şemasındaki tüm sütunlar korunur.
    """
    print(f"\n[3/3] Hava durumu birleştiriliyor: {path}")

    if not Path(path).exists():
        print(f"  [UYARI] {path} bulunamadı, hava sütunları 0 doldurulacak.")
        for col in HAVA_COLS:
            trafik_df[col] = 0
        return trafik_df

    hava = pd.read_parquet(path)

    if "date_hour" not in hava.columns:
        if "timestamp" in hava.columns:
            hava["timestamp"] = pd.to_datetime(hava["timestamp"])
            hava["date_hour"] = hava["timestamp"].dt.strftime("%Y-%m-%d %H")

    mevcut_hava_cols = [c for c in HAVA_COLS if c in hava.columns]
    keep      = ["date_hour", "geohash"] + mevcut_hava_cols
    hava_slim = hava[[c for c in keep if c in hava.columns]].copy()

    # geohash + date_hour ile join
    if "geohash" in hava_slim.columns and "geohash" in trafik_df.columns:
        hava_by_geo = (
            hava_slim
            .groupby(["geohash", "date_hour"])[mevcut_hava_cols]
            .mean()
            .reset_index()
        )
        merged    = trafik_df.merge(hava_by_geo, on=["geohash", "date_hour"], how="left")
        eslesen   = merged["temperature_c"].notna().sum() if "temperature_c" in merged.columns else 0
        print(f"  geohash + date_hour ile {eslesen:,} satır eşleşti")
    else:
        merged  = trafik_df.copy()
        eslesen = 0

    # Fallback: sadece date_hour
    eksik = merged["temperature_c"].isna().sum() if "temperature_c" in merged.columns else len(merged)
    if eksik > 0:
        hava_by_hour = (
            hava_slim
            .groupby("date_hour")[mevcut_hava_cols]
            .mean()
            .reset_index()
        )
        merged2 = merged.merge(hava_by_hour, on="date_hour", how="left", suffixes=("", "_fb"))
        for col in mevcut_hava_cols:
            fb = col + "_fb"
            if fb in merged2.columns:
                merged2[col] = merged2[col].fillna(merged2[fb])
                merged2 = merged2.drop(columns=[fb])
        merged = merged2
        print(f"  date_hour fallback ile {eksik:,} eksik satır dolduruldu")

    for col in mevcut_hava_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # DB'de olan ama hava parquet'te olmayan sütunları doldur
    for col in HAVA_COLS:
        if col not in merged.columns:
            merged[col] = 0

    toplam_eslesen  = merged["temperature_c"].ne(0).sum() if "temperature_c" in merged.columns else 0
    hava_eslesmeyen = len(merged) - int(toplam_eslesen)
    print(f"  Toplam eşleşen: {toplam_eslesen:,} / {len(merged):,} satır ✓")
    merged.attrs["hava_eslesmeyen"] = hava_eslesmeyen
    return merged


# =============================================================================
# 4. SON TEMİZLİK — DB şemasına uygun final
# =============================================================================

def finalize(df: pd.DataFrame, sample_n: int = None) -> pd.DataFrame:
    print(f"\n[Finalize] Son düzenleme yapılıyor...")
    print(f"  Gelen satır: {len(df):,}")

    # DB şemasındaki sütunları öncelikli al
    keep  = [c for c in LSTM_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in keep and
             c not in ["ym", "location", "date", "hour",
                       "timestamp_saat", "datetime_saat", "density"]]
    df = df[keep + extra]

    # Kritik NaN kontrolü — DB'nin zorunlu sütunları
    critical = [c for c in ["congestion_ratio", "datetime", "lat", "lon", "geohash"]
                if c in df.columns]
    before           = len(df)
    df               = df.dropna(subset=critical)
    finalize_silinen = before - len(df)
    print(f"  NaN atıldı: {finalize_silinen:,} satır")
    df.attrs["finalize_nan_silinen"] = finalize_silinen

    # Boolean sütunları düzelt
    bool_cols = ["is_weekend", "is_event_time", "road_closure",
                 "is_rainy", "is_snowy", "is_stormy", "is_foggy", "is_bad_weather"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    df = df.sort_values("datetime").reset_index(drop=True)

    if sample_n and len(df) > sample_n:
        step = len(df) // sample_n
        df   = df.iloc[::step].head(sample_n).reset_index(drop=True)
        print(f"  Örnekleme: {sample_n:,} satır (her {step}. satır)")

    print(f"  Final satır : {len(df):,}")
    print(f"  Final sütun : {len(df.columns)}")
    return df


# =============================================================================
# KALİTE RAPORU
# =============================================================================

def save_quality_report(df: pd.DataFrame, sayaclar: dict, out_path: str) -> None:
    rapor = {
        "olusturma_zamani": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "db_tablolari": {
            "LSTM_DESTEKLİ_İBB_BEŞİKTAŞ_VERİSİ": [c for c in TRAFIK_COLS if c in df.columns],
            "HavaDurumuVerisi":                    [c for c in HAVA_COLS   if c in df.columns],
            "EtkinlikVerisi":                      "etkinlik_kalite.json'dan okunur",
        },
        "final_veri": {
            "satir_sayisi":      int(len(df)),
            "sutun_sayisi":      int(len(df.columns)),
            "tarih_araligi": {
                "baslangic": str(df["date_hour"].min())[:10] if "date_hour" in df.columns else "",
                "bitis":     str(df["date_hour"].max())[:10] if "date_hour" in df.columns else "",
            },
            "benzersiz_geohash": int(df["geohash"].nunique()) if "geohash" in df.columns else 0,
            "ort_congestion":    round(float(df["congestion_ratio"].mean()), 4) if "congestion_ratio" in df.columns else 0,
        },
        "birlestirme_sayaclari": sayaclar,
        "kaynak_raporlar": {},
    }

    for dosya, anahtar in [
        ("trafik_kalite.json",             "trafik"),
        ("etkinlik_kalite.json",           "etkinlik"),
        ("gecmis_hava_durumu_kalite.json", "hava"),
    ]:
        if Path(dosya).exists():
            with open(dosya, encoding="utf-8") as f:
                rapor["kaynak_raporlar"][anahtar] = json.load(f)
        else:
            rapor["kaynak_raporlar"][anahtar] = {"durum": "rapor bulunamadi"}

    with open("veri_kalite_raporu.json", "w", encoding="utf-8") as f:
        json.dump(rapor, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  VERİ KALİTE RAPORU")
    print(f"{'='*60}")
    print(f"  Final satır          : {len(df):,}")
    s = sayaclar
    print(f"  Lag NaN silinen      : {s.get('lag_nan_silinen', 0):,}")
    print(f"  Finalize NaN silinen : {s.get('finalize_nan_silinen', 0):,}")
    print(f"  Hava eşleşmeyen→0   : {s.get('hava_eslesmeyen', 0):,}")
    print(f"  Etkinlik olan satır  : {s.get('etkinlik_icerisinde', 0):,}")
    print(f"  Kaydedildi           : veri_kalite_raporu.json")
    print(f"{'='*60}")


# =============================================================================
# ÖZET RAPOR
# =============================================================================

def print_summary(df: pd.DataFrame, out_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"  BİRLEŞTİRME TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  Satır sayısı     : {len(df):,}")
    print(f"  Sütun sayısı     : {len(df.columns)}")

    date_col = "date_hour" if "date_hour" in df.columns else "datetime"
    print(f"  Tarih aralığı    : {str(df[date_col].min())[:10]} "
          f"→ {str(df[date_col].max())[:10]}")

    if "congestion_ratio" in df.columns:
        print(f"  Ort. congestion  : {df['congestion_ratio'].mean():.3f}")
        print(f"  Max congestion   : {df['congestion_ratio'].max():.3f}")

    if "geohash" in df.columns:
        print(f"  Benzersiz geohash: {df['geohash'].nunique()}")

    if "is_event_time" in df.columns:
        print(f"  Etkinlik oranı   : %{df['is_event_time'].mean()*100:.1f}")

    if "is_bad_weather" in df.columns:
        print(f"  Kötü hava oranı  : %{df['is_bad_weather'].mean()*100:.1f}")

    print(f"\n  DB sütun uyumu:")
    for tablo, cols in [
        ("LSTM_DESTEKLİ_İBB", TRAFIK_COLS),
        ("HavaDurumuVerisi",   HAVA_COLS),
    ]:
        mevcut = [c for c in cols if c in df.columns]
        eksik  = [c for c in cols if c not in df.columns]
        print(f"    {tablo}: {len(mevcut)}/{len(cols)} sütun ✓"
              + (f" | eksik: {eksik}" if eksik else ""))

    print(f"\n  Çıktı: {out_path}")
    print(f"{'='*60}")


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Beşiktaş LSTM Eğitim Dataseti Birleştirici (DB şeması uyumlu)"
    )
    parser.add_argument("--trafik",   default=TRAFIK_FILE)
    parser.add_argument("--etkinlik", default=ETKINLIK_FILE)
    parser.add_argument("--hava",     default=HAVA_FILE)
    parser.add_argument("--out",      default=OUTPUT_FILE)
    parser.add_argument("--sample",   type=int, default=None)
    args = parser.parse_args()

    df = load_trafik(args.trafik)
    df = load_etkinlik(args.etkinlik, df)
    df = load_hava(args.hava, df)
    df = finalize(df, sample_n=args.sample)

    sayaclar = {
        "lag_nan_silinen":      df.attrs.get("lag_nan_silinen", 0),
        "finalize_nan_silinen": df.attrs.get("finalize_nan_silinen", 0),
        "hava_eslesmeyen":      df.attrs.get("hava_eslesmeyen", 0),
        "etkinlik_disinda":     int((df["is_event"] == 0).sum()) if "is_event" in df.columns else 0,
        "etkinlik_icerisinde":  int((df["is_event"] == 1).sum()) if "is_event" in df.columns else 0,
    }

    df.to_parquet(args.out, index=False, engine="pyarrow")
    print_summary(df, args.out)
    save_quality_report(df, sayaclar, args.out)

    print(f"\n  Sonraki adım: python lstm_egitim.py --data {args.out}")


if __name__ == "__main__":
    main()
