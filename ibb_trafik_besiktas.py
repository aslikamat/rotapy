"""
İBB Saatlik Trafik Yoğunluk Verisi — Beşiktaş Filtresi & LSTM Hazırlayıcı
===========================================================================
Ocak 2020 – Ocak 2025 arası 61 aylık İBB trafik verisini indirir,
Beşiktaş'a ait kayıtları filtreler ve LSTM modeline hazır tek bir
CSV dosyası üretir.

Kaynak: https://data.ibb.gov.tr/dataset/hourly-traffic-density-data-set
Lisans: İBB Açık Veri Lisansı (akademik kullanım serbest)

Kurulum:
    pip install requests pandas

Kullanım:
    python ibb_trafik_besiktas.py               # tüm ayları indir
    python ibb_trafik_besiktas.py --quick       # sadece 2023-2024 (hızlı test)
    python ibb_trafik_besiktas.py --sample      # 1 ay indir, yapıyı gör

Çıktı:
    besiktas_trafik_lstm.csv   → LSTM'e hazır ana dosya
    indirilen/                 → ham aylık CSV'ler (yedek)
"""

import requests
import pandas as pd
import os
import time
import argparse
from pathlib import Path
from io import StringIO

# =============================================================================
# İNDİRİLECEK DOSYALAR — Ocak 2020 → Ocak 2025
# =============================================================================

BASE_URL = (
    "https://data.ibb.gov.tr/dataset/"
    "3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/"
    "{resource_id}/download/traffic_density_{ym}.csv"
)

# resource_id'ler İBB portalından alındı (sayfa kaynak kodu)
MONTHLY_FILES = [
    ("202001", "db9c7fb3-e7f9-435a-92f4-1b917e357821"),
    ("202002", "5fb30ee1-e079-4865-a8cd-16efe2be8352"),
    ("202003", "efff9df8-4f40-4a46-8c99-2b3b4c5e2b8c"),
    ("202004", "9ead7895-27fb-4aed-847f-ffe1504c36fa"),
    ("202005", "5c0da73a-2fd6-4f98-90fe-aa32ce98b607"),
    ("202006", "62099013-e557-4d23-a2c0-70f7ee89c3b9"),
    ("202007", "e5fb99b3-afa0-4a9d-9bc8-cf98940da082"),
    ("202008", "dc40309d-7fd6-43e2-ad85-5db9db133a5b"),
    ("202009", "ef34bd55-86d8-4459-a710-79de30a45be2"),
    ("202010", "949d4a3b-91d2-4c56-b82f-4ef081e39c45"),
    ("202011", "93f996f1-70da-4500-951a-693c7e7066f6"),
    ("202012", "3e3161d8-7668-4694-829c-9179b41a775b"),
    ("202101", "fb7094a3-cf2f-46a6-996a-f6a9c5f3b9be"),
    ("202102", "395811ac-4152-4e04-88ef-8d4e30e6ac17"),
    ("202103", "fdbc8e2f-0cf1-4952-b50f-df8f40d5a649"),
    ("202104", "1eb158e8-8da7-4572-9825-108714a8856e"),
    ("202105", "00d72836-d035-462d-a66e-408883216195"),
    ("202106", "936faaf6-45ed-4463-ac57-85658c745cdc"),
    ("202107", "dde8cd53-f6aa-443e-916e-ab62a75be9a1"),
    ("202108", "345b86b6-15ea-4416-831a-478f0d6f9b19"),
    ("202109", "2bd92b0f-cbee-4cfb-9e94-c74b30c80fa2"),
    ("202110", "431bdb72-2204-4032-a96a-a810a2e88a0f"),
    ("202111", "b9131a98-99eb-4870-8960-e5f58f82e350"),
    ("202112", "2536eb25-9129-41e3-a028-f8a71fb16561"),
    ("202201", "8f492f69-95d0-46d7-b265-c141f8dba1a2"),
    ("202202", "7f655821-af63-4ba7-b3fe-9255a42ccff6"),
    ("202203", "3b7047b3-5b13-41c6-81d4-5dcf5c8c3696"),
    ("202204", "d57f1256-a0a5-4265-83b4-e06ee0458f49"),
    ("202205", "a250bd0a-ef49-4daf-a861-5a616056a9f4"),
    ("202206", "21a42752-4189-44fe-89ae-f17944f53a69"),
    ("202207", "287e7fc9-6d92-4019-ac58-ff6bca6e6151"),
    ("202208", "acd85951-6d23-4b50-bac6-d941f92af1ad"),
    ("202209", "a5da03fe-4a89-493b-ae60-aeb132511be9"),
    ("202210", "72183a60-d47f-4dc9-b1dc-fced0649dcf5"),
    ("202211", "7f463362-a580-41d9-a86a-a542818e7542"),
    ("202212", "dc788908-2b75-434f-9f3f-ef82ff33a158"),
    ("202301", "42fa7a5f-29f1-4b38-9dfa-ac7c8fe3c77d"),
    ("202302", "366befd8-defd-4f79-a3d2-0e7948c649ff"),
    ("202303", "6a60b03a-bf25-4575-9dce-e21fe0e04e77"),
    ("202304", "ce65562e-0d17-4d7e-8090-9484990a8f2b"),
    ("202305", "d0a71c11-47d2-4f98-8745-c9446b10bf18"),
    ("202306", "a99913df-dccc-4b7d-b6e3-963ccb5d27b1"),
    ("202307", "3de18c1e-57c0-4493-9b75-5a896edae0ff"),
    ("202308", "f6a1e2d7-0d9f-4d84-90c6-2729a0869308"),
    ("202309", "7b9a35a7-dc9c-4044-b117-1c0003104630"),
    ("202310", "342488a2-a00f-4ba7-bb4a-345f75f1120d"),
    ("202311", "e6a18077-2bd9-4201-8d4a-5398b0e2d99c"),
    ("202312", "aa58374d-ef6f-411f-8271-5b63eefe4fde"),
    ("202401", "7d9cbf11-f4b8-464d-bb3a-642b79e8b32b"),
    ("202402", "601cd734-9a62-44e0-89e5-bbfc2161d389"),
    ("202403", "b67e9415-0ba8-4319-8d36-359240a93808"),
    ("202404", "0c7d60f3-8349-4836-a1c2-56ec93cbbd50"),
    ("202405", "674604c8-8d08-42ff-a0b3-e2bde9f39455"),
    ("202406", "674ba2c5-76b0-4f24-8e17-9aa2071d2572"),
    ("202407", "0019216e-48e0-4cec-9ab8-93d67f66dac3"),
    ("202408", "168467fe-0495-4cdf-a93a-7c1e91179457"),
    ("202409", "914cb0b9-d941-4408-98eb-f378519c26f4"),
    ("202410", "d291989c-429d-4e61-9c70-1f76294b96b8"),
    ("202411", "bedd5ab2-9a00-4966-9921-9672d4478a51"),
    ("202412", "76671ebe-2fd2-426f-b85a-e3772263f483"),
    ("202501", "57cb067b-1a0b-460b-8342-7884bd4537e8"),
]

# Sadece 2023-2024 (--quick modu)
QUICK_FILES = [f for f in MONTHLY_FILES if f[0].startswith(("2023", "2024"))]

# =============================================================================
# BEŞİKTAŞ FİLTRESİ
# Veri setindeki GEOFENCE_LAT / GEOFENCE_LON veya DISTRICT sütununa göre
# =============================================================================

# Beşiktaş bbox — bu koordinatlar içindeki kayıtlar alınır
BESIKTAS_BBOX = {
    "lat_min": 41.025,
    "lat_max": 41.075,
    "lon_min": 28.980,
    "lon_max": 29.040,
}

# Beşiktaş ile ilişkili string etiketler (DISTRICT/LOCATION sütunu varsa)
BESIKTAS_KEYWORDS = [
    "beşiktaş", "besiktas", "barbaros", "çırağan", "ciragan",
    "vodafone", "zorlu", "levent", "balmumcu", "ortaköy", "ortakoy",
    "bebek", "kuruçeşme", "kurucesme", "akaretler", "ihlamur",
    "yıldız", "yildiz", "dikilitaş",
]


def is_besiktas_row(row: pd.Series, lat_col: str, lon_col: str) -> bool:
    """Koordinat veya isim bazlı Beşiktaş filtresi."""
    # Koordinat bazlı (öncelikli)
    try:
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        bb = BESIKTAS_BBOX
        if bb["lat_min"] <= lat <= bb["lat_max"] and \
           bb["lon_min"] <= lon <= bb["lon_max"]:
            return True
    except (ValueError, TypeError):
        pass

    # İsim bazlı (yedek)
    for col in row.index:
        val = str(row[col]).lower()
        if any(kw in val for kw in BESIKTAS_KEYWORDS):
            return True

    return False


# =============================================================================
# VERİ İNDİRME
# =============================================================================

def download_month(ym: str, resource_id: str, cache_dir: Path) -> pd.DataFrame | None:
    """
    Tek aylık CSV'yi indirir (önce cache'e bakar).
    Dönen DataFrame ham veridir, henüz filtrelenmemiş.
    """
    cache_file = cache_dir / f"traffic_density_{ym}.csv"

    # Cache varsa yeniden indirme
    if cache_file.exists():
        try:
            return pd.read_csv(cache_file, low_memory=False)
        except Exception:
            pass

    url = BASE_URL.format(resource_id=resource_id, ym=ym)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        # Cache'e kaydet
        cache_file.write_bytes(r.content)
        return pd.read_csv(StringIO(r.text), low_memory=False)
    except requests.exceptions.Timeout:
        print(f"    [TIMEOUT] {ym}")
        return None
    except Exception as e:
        print(f"    [HATA] {ym}: {e}")
        return None


# =============================================================================
# SÜTUN TESPITI
# İBB farklı dönemlerde farklı sütun adları kullanmış olabilir
# =============================================================================

COLUMN_ALIASES = {
    "lat":        ["LATITUDE", "LAT", "latitude", "lat", "GEOFENCE_LAT",
                   "ENLEM", "enlem"],
    "lon":        ["LONGITUDE", "LON", "longitude", "lon", "GEOFENCE_LON",
                   "BOYLAM", "boylam"],
    "datetime":   ["DATE_TIME", "DATETIME", "date_time", "datetime",
                   "MEASUREMENT_TIME", "TARIH_SAAT", "tarih_saat"],
    "speed":      ["SPEED", "speed", "HIZ", "hiz", "AVG_SPEED",
                   "AVERAGE_SPEED"],
    "density":    ["NUMBER_OF_VEHICLES", "DENSITY", "density",
                   "ARAC_SAYISI", "VEHICLE_COUNT", "COUNT",
                   "MINIMUM_SPEED", "NUMBER_OF_VEHICLES_ANALYZED"],
    "location":   ["GEOFENCE_NAME", "LOCATION", "DISTRICT", "location",
                   "BOLGE", "ROAD_NAME"],
}


def detect_col(df: pd.DataFrame, key: str) -> str | None:
    """Sütun adı tahmin et."""
    for alias in COLUMN_ALIASES.get(key, []):
        if alias in df.columns:
            return alias
    # Kısmi eşleşme
    key_lower = key.lower()
    for col in df.columns:
        if key_lower in col.lower():
            return col
    return None


# =============================================================================
# TEK AYLIK VERİYİ İŞLE
# =============================================================================

def process_month(df: pd.DataFrame, ym: str) -> pd.DataFrame:
    """
    Ham aylık veriyi alır:
      1. Sütunları tespit et
      2. Beşiktaş filtrele
      3. LSTM feature'larını ekle
      4. Standart sütun adlarıyla döndür
    """
    lat_col  = detect_col(df, "lat")
    lon_col  = detect_col(df, "lon")
    dt_col   = detect_col(df, "datetime")
    spd_col  = detect_col(df, "speed")
    den_col  = detect_col(df, "density")
    loc_col  = detect_col(df, "location")

    # Koordinat sütunu yoksa filtreleme yapamayız
    if not lat_col or not lon_col:
        # Yine de isim bazlı deneyelim
        mask = df.apply(
            lambda r: any(
                kw in str(r.values).lower() for kw in BESIKTAS_KEYWORDS
            ), axis=1
        )
        df_b = df[mask].copy()
    else:
        mask = df.apply(
            lambda r: is_besiktas_row(r, lat_col, lon_col), axis=1
        )
        df_b = df[mask].copy()

    if df_b.empty:
        return pd.DataFrame()

    # Standart sütunlar
    out = pd.DataFrame()
    out["ym"]       = ym

    # Tarih/saat
    if dt_col:
        out["datetime"] = pd.to_datetime(df_b[dt_col], errors="coerce")
    else:
        # Tarih yok — ay bilgisinden üret
        out["datetime"] = pd.to_datetime(f"{ym[:4]}-{ym[4:]}-01")

    out["date"]     = out["datetime"].dt.date.astype(str)
    out["hour"]     = out["datetime"].dt.hour
    out["weekday"]  = out["datetime"].dt.weekday   # 0=Pzt, 6=Paz
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)

    # Konum
    out["lat"]      = pd.to_numeric(df_b[lat_col],  errors="coerce") if lat_col  else None
    out["lon"]      = pd.to_numeric(df_b[lon_col],  errors="coerce") if lon_col  else None
    out["location"] = df_b[loc_col].values if loc_col else "Beşiktaş"

    # Trafik metrikleri
    out["speed"]    = pd.to_numeric(df_b[spd_col],  errors="coerce") if spd_col  else None
    out["density"]  = pd.to_numeric(df_b[den_col],  errors="coerce") if den_col  else None

    # Orijinal sütunların tamamını da ekle (bilgi kaybı olmasın)
    for col in df_b.columns:
        if col not in [lat_col, lon_col, dt_col, spd_col, den_col, loc_col]:
            safe = col.lower().replace(" ", "_")
            if safe not in out.columns:
                out[safe] = df_b[col].values

    return out.reset_index(drop=True)


# =============================================================================
# LSTM FEATURE MÜHENDİSLİĞİ
# =============================================================================

def add_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    LSTM için ek zaman serisi özellikler ekler:
      - Saatin sinüs/kosinüs dönüşümü (döngüsel zaman)
      - Haftanın günü sinüs/kosinüs
      - Önceki saatin yoğunluğu (lag_1h)
      - 24 saat önceki yoğunluk (lag_24h)
      - 7 gün önceki aynı saat (lag_7d)
      - 3 saatlik hareketli ortalama
    """
    import numpy as np

    df = df.sort_values("datetime").reset_index(drop=True)

    # Döngüsel zaman encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"]  = np.sin(2 * np.pi * df["weekday"] / 7)
    df["day_cos"]  = np.cos(2 * np.pi * df["weekday"] / 7)

    # Hedef değişken olarak density veya speed kullan
    target_col = "density" if "density" in df.columns and df["density"].notna().any() \
                 else "speed"

    if target_col in df.columns:
        df["target"] = df[target_col]
        # Lag özellikleri (lokasyon bazında)
        for loc, grp in df.groupby("location"):
            idx = grp.index
            df.loc[idx, "lag_1h"]  = grp[target_col].shift(1).values
            df.loc[idx, "lag_24h"] = grp[target_col].shift(24).values
            df.loc[idx, "lag_7d"]  = grp[target_col].shift(24 * 7).values
            df.loc[idx, "ma_3h"]   = grp[target_col].rolling(3, min_periods=1).mean().values

    return df


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="İBB Trafik Verisi — Beşiktaş Filtresi & LSTM Hazırlayıcı"
    )
    parser.add_argument("--quick",  action="store_true",
                        help="Sadece 2023-2024 verilerini indir (hızlı test)")
    parser.add_argument("--sample", action="store_true",
                        help="Sadece 1 ay indir, veri yapısını göster")
    parser.add_argument("--no-cache", action="store_true",
                        help="Cache'i yok say, hepsini yeniden indir")
    parser.add_argument("--out", default="besiktas_trafik_lstm.csv",
                        help="Çıktı dosya adı")
    args = parser.parse_args()

    cache_dir = Path("indirilen")
    cache_dir.mkdir(exist_ok=True)

    if args.no_cache:
        for f in cache_dir.glob("*.csv"):
            f.unlink()

    files = MONTHLY_FILES
    if args.quick:
        files = QUICK_FILES
        print(f"[Hızlı mod] Sadece 2023-2024: {len(files)} ay")
    elif args.sample:
        files = [MONTHLY_FILES[-1]]  # En son ay
        print(f"[Örnek mod] Tek ay: {files[0][0]}")
    else:
        print(f"[Tam mod] {len(files)} ay, Ocak 2020 → Ocak 2025")

    print(f"Cache dizini: {cache_dir}/")
    print(f"Çıktı: {args.out}\n")

    all_frames = []
    for i, (ym, resource_id) in enumerate(files, 1):
        print(f"[{i:>2}/{len(files)}] {ym} indiriliyor...", end=" ")
        df_raw = download_month(ym, resource_id, cache_dir)

        if df_raw is None:
            print("ATLANДИ")
            continue

        print(f"{len(df_raw):>8,} satır ham →", end=" ")
        df_b = process_month(df_raw, ym)

        if df_b.empty:
            print("Beşiktaş kaydı YOK")
            # Sütunları göster — filtreleme sorununu anlamak için
            print(f"          Sütunlar: {list(df_raw.columns[:8])}")
        else:
            print(f"{len(df_b):>6,} Beşiktaş kaydı ✓")
            all_frames.append(df_b)

        time.sleep(0.3)  # İBB sunucusuna nazik ol

    if not all_frames:
        print("\n[SONUÇ] Hiç Beşiktaş kaydı bulunamadı.")
        print("Olası sebep: veri setinde 'DISTRICT' veya koordinat sütunu yok.")
        print("→ İBB ile iletişime geçin: data.ibb.gov.tr/contact")
        return

    print(f"\n[Birleştirme] {len(all_frames)} ay birleştiriliyor...")
    merged = pd.concat(all_frames, ignore_index=True)

    # Tarihe göre sırala
    if "datetime" in merged.columns:
        merged = merged.sort_values("datetime")

    # Eksik veri temizleme ve doldurma
    target = "density" if "density" in merged.columns and merged["density"].notna().any() else "speed"
    print(f"\n[Temizlik] Eksik veri dolduruluyor (hedef: {target})...")
    merged = clean_and_fill(merged, target_col=target)

    # LSTM feature'ları ekle
    print("\n[Feature mühendisliği] Lag ve döngüsel özellikler ekleniyor...")
    merged = add_lstm_features(merged)

    # Kaydet
    merged.to_csv(args.out, index=False, encoding="utf-8-sig")

    # Özet
    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  Toplam satır  : {len(merged):,}")
    print(f"  Tarih aralığı : {merged['date'].min()} → {merged['date'].max()}")
    print(f"  Sütunlar      : {list(merged.columns)}")
    print(f"  Çıktı         : {args.out}")
    print(f"{'='*60}")

    # Sample göster
    print("\nİlk 5 satır:")
    show_cols = ["datetime", "location", "lat", "lon",
                 "speed", "density", "hour", "is_weekend", "target"]
    show_cols = [c for c in show_cols if c in merged.columns]
    print(merged[show_cols].head().to_string(index=False))





# =============================================================================
# EKSİK VERİ TEMİZLEME VE DOLDURMA
# =============================================================================

def clean_and_fill(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """
    1. Tüm 24 saati koru (gece verisi LSTM için önemli)
    2. 1-2 saatlik eksiklikleri aynı weekday+hour ortalamasıyla doldur
    3. 3+ ardışık eksik saati at
    """
    import numpy as np

    print(f"  Temizlik öncesi : {len(df):,} satır")

    if target_col not in df.columns or df[target_col].isna().sum() == 0:
        print(f"  Eksik değer yok, doldurma atlandı.")
        return df

    eksik_once = df[target_col].isna().sum()

    # 2. Weekday + hour ortalaması tablosu
    ortalama = (
        df.groupby(["weekday", "hour", "location"])[target_col]
        .mean()
        .rename("ortalama")
        .reset_index()
    )
    df = df.merge(ortalama, on=["weekday", "hour", "location"], how="left")

    # 3. Ardışık eksik bloklarını tespit et (lokasyon bazında)
    filled = []
    for loc, grp in df.groupby("location"):
        grp = grp.sort_values("datetime").copy()

        # Ardışık eksik sayacı
        grp["eksik_ardisik"] = (
            grp[target_col].isna()
            .astype(int)
            .groupby((~grp[target_col].isna()).cumsum())
            .cumsum()
        )

        # 3+ ardışık eksik → at
        grp = grp[grp["eksik_ardisik"] < 3].copy()

        # 1-2 eksik → ortalama ile doldur
        mask_eksik = grp[target_col].isna()
        grp.loc[mask_eksik, target_col] = grp.loc[mask_eksik, "ortalama"]

        filled.append(grp)

    df = pd.concat(filled, ignore_index=True).drop(
        columns=["ortalama", "eksik_ardisik"], errors="ignore"
    )

    eksik_sonra = df[target_col].isna().sum()
    print(f"  Dolduruldu      : {eksik_once - eksik_sonra:,} eksik değer")
    print(f"  Hâlâ eksik      : {eksik_sonra:,} (3+ ardışık blok atıldı)")
    print(f"  Temizlik sonrası: {len(df):,} satır")

    return df
if __name__ == "__main__":
    main()
