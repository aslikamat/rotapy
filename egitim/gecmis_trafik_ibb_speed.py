"""
İBB Saatlik Trafik Yoğunluk Verisi — Beşiktaş Filtresi & LSTM Hazırlayıcı
===========================================================================
Ocak 2020 – Ocak 2025 arası 61 aylık İBB trafik verisini indirir,
Beşiktaş'a ait kayıtları filtreler ve LSTM modeline hazır tek bir
parquet dosyası üretir.

HEDEF DEĞİŞKEN (congestion_ratio) HIZDAN HESAPLANIR, araç sayısından DEĞİL:

    congestion_ratio = 1 - saatlik_ortalama_hız / serbest_akış_hızı     (0..1)
        0 = yol boş (hız serbest akış hızında), 1 = trafik durmuş

İBB verisinde serbest akış hızı yok. Her hücre (geohash[:6]) için saatlik ortalama
hızın q'luk yüzdeliği (varsayılan %95) yol boşken ulaşılan hız kabul edilir.
Bu yüzdelik, veri sızıntısı olmasın diye sadece zaman olarak ilk %70'lik kısımdan
(eğitim bölümüyle aynı oran) hesaplanır. TomTom canlı verisindeki
1 - currentSpeed/freeFlowSpeed tanımıyla aynı formüldür.

Kaynak: https://data.ibb.gov.tr/dataset/hourly-traffic-density-data-set
Lisans: İBB Açık Veri Lisansı (akademik kullanım serbest)

Kurulum:
    pip install requests pandas pyarrow

Kullanım:
    python gecmis_trafik_ibb.py               # tüm ayları indir
    python gecmis_trafik_ibb.py --quick       # sadece 2023-2024
    python gecmis_trafik_ibb.py --sample      # 1 ay indir, yapıyı gör
    python gecmis_trafik_ibb.py --quick --freeflow-q 1.0 --freeflow-oran 1.0
        # serbest akış = hücrenin 2023-2024'te gördüğü EN YÜKSEK saatlik ortalama hız
    python gecmis_trafik_ibb.py --quick --freeflow-q 1.0 --freeflow-oran 1.0 --freeflow-min-arac 20
        # aynısı, ama az araçlı (güvenilmez) saatler dışarıda

Çıktı:
    besiktas_trafik_lstm.parquet → LSTM'e hazır ana dosya
                                   (congestion_ratio, freeflow_speed, lag_*, ma_3h dahil)
    trafik_kalite.json           → veri kalite raporu
    indirilen/                   → ham aylık CSV'ler (cache)
"""

import requests
import pandas as pd
import numpy as np
import os
import time
import argparse
import json
from pathlib import Path
from io import StringIO

# =============================================================================
# GEOHASH
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
# İNDİRİLECEK DOSYALAR
# =============================================================================

BASE_URL = (
    "https://data.ibb.gov.tr/dataset/"
    "3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/"
    "{resource_id}/download/traffic_density_{ym}.csv"
)

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

QUICK_FILES = [f for f in MONTHLY_FILES if f[0].startswith(("2023", "2024"))]

# =============================================================================
# BEŞİKTAŞ FİLTRESİ
# =============================================================================

BESIKTAS_BBOX = {
    "lat_min": 41.025, "lat_max": 41.075,
    "lon_min": 28.980, "lon_max": 29.040,
}

# Boğaz'ın Asya yakasına taşan hücreler — eğitim verisine ALINMAZ.
# Neden: etkinlik etkisi düz mesafeyle hesaplanıyor; Üsküdar hücreleri Vodafone Park'ın
# 4.6 km etki alanında kalıyor ama Boğaz'ın karşı yakasında. Hava ve etkinlik verisi de
# sadece Beşiktaş için toplanıyor. (osm_nokta_sec.py'deki HARIC_HUCRELER ile aynı liste.)
HARIC_HUCRELER = (
    "sxk9kh", "sxk9kj", "sxk9kk", "sxk9km", "sxk9kn",
    "sxk9kq", "sxk9ks", "sxk9kt", "sxk9kw", "sxk9kx",
)

BESIKTAS_KEYWORDS = [
    "beşiktaş", "besiktas", "barbaros", "çırağan", "ciragan",
    "vodafone", "zorlu", "levent", "balmumcu", "ortaköy", "ortakoy",
    "bebek", "kuruçeşme", "kurucesme", "akaretler", "ihlamur",
    "yıldız", "yildiz", "dikilitaş",
]


def is_besiktas_row(row: pd.Series, lat_col: str, lon_col: str) -> bool:
    try:
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        bb  = BESIKTAS_BBOX
        if bb["lat_min"] <= lat <= bb["lat_max"] and \
           bb["lon_min"] <= lon <= bb["lon_max"]:
            return True
    except (ValueError, TypeError):
        pass
    for col in row.index:
        if any(kw in str(row[col]).lower() for kw in BESIKTAS_KEYWORDS):
            return True
    return False


# =============================================================================
# VERİ İNDİRME
# =============================================================================

def download_month(ym: str, resource_id: str, cache_dir: Path) -> pd.DataFrame | None:
    cache_file = cache_dir / f"traffic_density_{ym}.csv"
    if cache_file.exists():
        try:
            return pd.read_csv(cache_file, low_memory=False)
        except Exception:
            pass
    url = BASE_URL.format(resource_id=resource_id, ym=ym)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        cache_file.write_bytes(r.content)
        return pd.read_csv(StringIO(r.text), low_memory=False)
    except requests.exceptions.Timeout:
        print(f"    [TIMEOUT] {ym}")
        return None
    except Exception as e:
        print(f"    [HATA] {ym}: {e}")
        return None


# =============================================================================
# SÜTUN TESPİTİ
# =============================================================================

COLUMN_ALIASES = {
    "lat":      ["LATITUDE", "LAT", "latitude", "lat", "GEOFENCE_LAT", "ENLEM", "enlem"],
    "lon":      ["LONGITUDE", "LON", "longitude", "lon", "GEOFENCE_LON", "BOYLAM", "boylam"],
    "datetime": ["DATE_TIME", "DATETIME", "date_time", "datetime",
                 "MEASUREMENT_TIME", "TARIH_SAAT", "tarih_saat"],
    "speed":    ["SPEED", "speed", "HIZ", "hiz", "AVG_SPEED", "AVERAGE_SPEED"],
    "density":  ["NUMBER_OF_VEHICLES", "DENSITY", "density", "ARAC_SAYISI",
                 "VEHICLE_COUNT", "COUNT", "MINIMUM_SPEED", "NUMBER_OF_VEHICLES_ANALYZED"],
    "location": ["GEOFENCE_NAME", "LOCATION", "DISTRICT", "location", "BOLGE", "ROAD_NAME"],
}


def detect_col(df: pd.DataFrame, key: str) -> str | None:
    for alias in COLUMN_ALIASES.get(key, []):
        if alias in df.columns:
            return alias
    for col in df.columns:
        if key.lower() in col.lower():
            return col
    return None


# =============================================================================
# TEK AYLIK VERİYİ İŞLE
# =============================================================================

def process_month(df: pd.DataFrame, ym: str) -> pd.DataFrame:
    lat_col = detect_col(df, "lat")
    lon_col = detect_col(df, "lon")
    dt_col  = detect_col(df, "datetime")
    spd_col = detect_col(df, "speed")
    den_col = detect_col(df, "density")
    loc_col = detect_col(df, "location")

    # ── Vektörize Beşiktaş filtresi — satır satır döngü YOK ─────────────────
    if lat_col and lon_col:
        lats = pd.to_numeric(df[lat_col], errors="coerce")
        lons = pd.to_numeric(df[lon_col], errors="coerce")
        bb   = BESIKTAS_BBOX
        mask = (
            lats.between(bb["lat_min"], bb["lat_max"]) &
            lons.between(bb["lon_min"], bb["lon_max"])
        )
        # Koordinat eşleşmeyenleri isim bazlı dene (küçük subset)
        no_coord = mask[~mask].index
        if len(no_coord) > 0 and loc_col:
            loc_mask = df.loc[no_coord, loc_col].astype(str).str.lower().apply(
                lambda v: any(kw in v for kw in BESIKTAS_KEYWORDS)
            )
            mask.loc[no_coord[loc_mask]] = True
    else:
        # Koordinat yok — sadece isim bazlı (yavaş ama nadir durum)
        mask = df.apply(
            lambda r: any(kw in str(r.values).lower() for kw in BESIKTAS_KEYWORDS),
            axis=1
        )

    df_b = df[mask].copy()
    if df_b.empty:
        return pd.DataFrame()

    out = pd.DataFrame()

    # datetime
    if dt_col:
        out["datetime"] = pd.to_datetime(df_b[dt_col].values, errors="coerce")
    else:
        out["datetime"] = pd.to_datetime(f"{ym[:4]}-{ym[4:]}-01")

    # date_hour
    out["date_hour"]  = out["datetime"].dt.strftime("%Y-%m-%d %H")
    out["weekday"]    = out["datetime"].dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)

    # Konum + geohash — vektörize
    lat_vals = pd.to_numeric(df_b[lat_col].values, errors="coerce") if lat_col else None
    lon_vals = pd.to_numeric(df_b[lon_col].values, errors="coerce") if lon_col else None
    out["lat"] = lat_vals
    out["lon"] = lon_vals

    if lat_vals is not None and lon_vals is not None:
        out["geohash"] = [
            encode_geohash(la, lo) if (la == la and lo == lo) else ""
            for la, lo in zip(lat_vals, lon_vals)
        ]
    else:
        out["geohash"] = ""

    # Trafik
    out["speed"]   = pd.to_numeric(df_b[spd_col].values, errors="coerce") if spd_col else None
    out["density"] = pd.to_numeric(df_b[den_col].values, errors="coerce") if den_col else None

    # Ek sütunlar
    for col in df_b.columns:
        if col not in [lat_col, lon_col, dt_col, spd_col, den_col, loc_col]:
            safe = col.lower().replace(" ", "_")
            if safe not in out.columns:
                out[safe] = df_b[col].values

    return out.reset_index(drop=True)


# =============================================================================
# EKSİK VERİ TEMİZLEME
# =============================================================================

def clean_and_fill(df: pd.DataFrame, target_col: str = "target") -> tuple:
    """
    1. 1-2 saatlik eksiklikleri weekday + geohash ortalamasıyla doldur
    2. 3+ ardışık eksik saati at
    Tuple döner: (df, sayac_dict)
    """
    print(f"  Temizlik öncesi : {len(df):,} satır")

    if target_col not in df.columns or df[target_col].isna().sum() == 0:
        print(f"  Eksik değer yok, doldurma atlandı.")
        return df, {"eksik_once": 0, "dolduruldu": 0, "kalan_eksik": 0}

    eksik_once = df[target_col].isna().sum()
    group_col  = "geohash" if "geohash" in df.columns else "lat"

    ortalama = (
        df.groupby(["weekday", group_col])[target_col]
        .mean()
        .rename("ortalama")
        .reset_index()
    )
    df = df.merge(ortalama, on=["weekday", group_col], how="left")

    filled = []
    for loc, grp in df.groupby(group_col):
        grp = grp.sort_values("datetime").copy()
        grp["eksik_ardisik"] = (
            grp[target_col].isna()
            .astype(int)
            .groupby((~grp[target_col].isna()).cumsum())
            .cumsum()
        )
        grp = grp[grp["eksik_ardisik"] < 3].copy()
        mask_eksik = grp[target_col].isna()
        grp.loc[mask_eksik, target_col] = grp.loc[mask_eksik, "ortalama"]
        filled.append(grp)

    df = pd.concat(filled, ignore_index=True).drop(
        columns=["ortalama", "eksik_ardisik"], errors="ignore"
    )

    eksik_sonra = df[target_col].isna().sum()
    dolduruldu  = eksik_once - eksik_sonra

    print(f"  Dolduruldu      : {dolduruldu:,} eksik değer (ortalama ile)")
    print(f"  Hâlâ eksik      : {eksik_sonra:,} (3+ ardışık blok atıldı)")
    print(f"  Temizlik sonrası: {len(df):,} satır")

    return df, {
        "eksik_once":  int(eksik_once),
        "dolduruldu":  int(dolduruldu),
        "kalan_eksik": int(eksik_sonra),
    }


# =============================================================================
# HIZDAN CONGESTION_RATIO
# =============================================================================

def add_congestion_from_speed(df: pd.DataFrame, q: float = 0.95, ratio: float = 0.70,
                              min_rows: int = 100, min_arac: int = 0) -> tuple:
    """
    congestion_ratio = 1 - speed / freeflow_speed, 0..1 aralığına kırpılır.

    speed          : İBB AVERAGE_SPEED (hücredeki tüm araçların saatlik ortalama hızı)
    freeflow_speed : hücre (geohash[:6]) başına speed'in q'luk yüzdeliği.
                     q=1.0 → hücrenin gördüğü EN YÜKSEK saatlik ortalama hız.
                     Sadece zamanın ilk `ratio` kadarlık kısmından hesaplanır
                     (ratio=1.0 → tüm veri). Referans döneminde min_rows'dan az satırı
                     olan hücrede tüm hücrelerin ortanca serbest akış hızı kullanılır.
    min_arac       : >0 ise serbest akış hızı sadece NUMBER_OF_VEHICLES >= min_arac olan
                     saatlerden hesaplanır. Araç sayısı çok düşük saatlerde ortalama hız
                     tek bir aracın hızına eşit olabilir; en yüksek değeri (q=1.0) bu tür
                     sıradışı saatler belirler, bu filtre onları dışarıda bırakır.
    Dönen: (df, bilgi_sözlüğü) — bilgi kalite raporuna yazılır.
    """
    df = df.copy()
    hucre = df["geohash"].astype(str).str[:6]

    t = df["datetime"].sort_values().reset_index(drop=True)
    sinir = t.iloc[min(len(t) - 1, max(0, int(len(t) * ratio) - 1))]
    ref = df["datetime"] <= sinir
    if min_arac > 0:
        if "density" in df.columns:
            ref = ref & (df["density"] >= min_arac)
        else:
            print("  [UYARI] density (araç sayısı) sütunu yok, --freeflow-min-arac yok sayıldı.")
            min_arac = 0

    ff   = df.loc[ref, "speed"].groupby(hucre[ref]).quantile(q)
    adet = df.loc[ref, "speed"].groupby(hucre[ref]).count()
    ff   = ff[adet >= min_rows]
    if ff.empty:
        raise ValueError("Serbest akış hızı hesaplanamadı: hiçbir hücrede yeterli satır yok. "
                         "--freeflow-min-arac değerini düşürün ya da 0 yapın.")
    genel = float(ff.median())
    yedek_hucre = sorted(set(hucre.unique()) - set(ff.index))

    # Sıradışı değer teşhisi: en yüksek hız, %99'luk değerin çok üstünde mi?
    mx  = df.loc[ref, "speed"].groupby(hucre[ref]).max()
    q99 = df.loc[ref, "speed"].groupby(hucre[ref]).quantile(0.99)
    sapma = (mx / q99).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)

    df["freeflow_speed"] = hucre.map(ff).fillna(genel).round(2)
    oran = 1 - df["speed"] / df["freeflow_speed"].replace(0, np.nan)
    df["congestion_ratio"] = oran.clip(0, 1).round(4)

    once = len(df)
    df = df.dropna(subset=["congestion_ratio"]).reset_index(drop=True)
    saat_ort = df.groupby(df["datetime"].dt.hour)["congestion_ratio"].mean().round(3)

    bilgi = {
        "tanim":                     "1 - AVERAGE_SPEED / freeflow_speed (hücre bazlı yüzdelik)",
        "freeflow_yuzdelik":         q,
        "freeflow_referans_orani":   ratio,
        "freeflow_referans_sonu":    str(sinir),
        "freeflow_min_arac":         int(min_arac),
        "freeflow_hucre_sayisi":     int(len(ff)),
        "freeflow_yedek_hucreler":   yedek_hucre,
        "freeflow_min_ortanca_max":  [round(float(ff.min()), 1), round(genel, 1),
                                      round(float(ff.max()), 1)],
        "max_q99_orani_en_yuksek":   round(float(sapma.iloc[0]), 2) if len(sapma) else None,
        "nan_nedeniyle_silinen":     int(once - len(df)),
        "ortalama":                  round(float(df["congestion_ratio"].mean()), 4),
        "sifir_pct":                 round(float((df["congestion_ratio"] == 0).mean() * 100), 2),
        "bir_pct":                   round(float((df["congestion_ratio"] == 1).mean() * 100), 2),
        "saatlik_ortalama":          {int(h): float(v) for h, v in saat_ort.items()},
    }
    ad = "en yüksek değer" if q >= 1.0 else f"%{q * 100:.0f}'lik değer"
    print(f"  Serbest akış hızı (hücre bazlı {ad}"
          f"{f', araç sayısı >= {min_arac}' if min_arac else ''}): "
          f"min {bilgi['freeflow_min_ortanca_max'][0]} / ortanca {bilgi['freeflow_min_ortanca_max'][1]} "
          f"/ max {bilgi['freeflow_min_ortanca_max'][2]} km/s  "
          f"(referans: {str(sinir)[:10]} sonuna kadar)")
    print(f"  Ortalama congestion_ratio: {bilgi['ortalama']:.3f}  "
          f"(0'a kırpılan %{bilgi['sifir_pct']}, 1'e kırpılan %{bilgi['bir_pct']})")
    pik = saat_ort.sort_values(ascending=False).head(3)
    print("  En yoğun 3 saat: " + ", ".join(f"{int(h):02d}:00 → {v:.2f}" for h, v in pik.items()))
    kotu = sapma[sapma > 1.20]
    if len(kotu):
        print(f"  [UYARI] {len(kotu)} hücrede en yüksek hız, %99'luk değerin 1.2 katından fazla "
              f"(sıradışı değer şüphesi): "
              + ", ".join(f"{c} ({v:.2f}x)" for c, v in kotu.head(3).items()))
        if q >= 0.999 and not min_arac:
            print("          Öneri: --freeflow-min-arac 20 (ya da benzeri) ile az araçlı saatleri "
                  "dışarıda bırakın.")
    if yedek_hucre:
        print(f"  [UYARI] Referans dönemde az verili hücreler (ortanca kullanıldı): {yedek_hucre}")
    return df, bilgi


# =============================================================================
# LSTM FEATURE MÜHENDİSLİĞİ
# =============================================================================

def add_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Döngüsel zaman + lag özellikleri ekler.
    Lag'ler hedefle AYNI birimde olsun diye congestion_ratio'dan hesaplanır.
    hour sütunu YOK — datetime veya date_hour'dan türetilir.
    """
    df = df.sort_values("datetime").reset_index(drop=True)

    # Saati datetime'dan türet (hour sütunu artık yok)
    if "datetime" in df.columns:
        hour = df["datetime"].dt.hour
    else:
        hour = pd.to_datetime(df["date_hour"], format="%Y-%m-%d %H").dt.hour

    # Döngüsel zaman encoding
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"]  = np.sin(2 * np.pi * df["weekday"] / 7)
    df["day_cos"]  = np.cos(2 * np.pi * df["weekday"] / 7)

    # Hedef değişken: hıza bağlı congestion_ratio
    target_col = "congestion_ratio"
    df["target"] = df[target_col]
    group_col    = "geohash" if "geohash" in df.columns else "lat"
    for loc, grp in df.groupby(group_col):
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
    parser.add_argument("--quick",    action="store_true",
                        help="Sadece 2023-2024 (hızlı test)")
    parser.add_argument("--sample",   action="store_true",
                        help="Sadece 1 ay indir")
    parser.add_argument("--cache-dir", default="indirilen",
                        help="Ham aylık CSV'lerin bulunduğu klasör (varsayılan: bulunduğunuz "
                             "klasördeki 'indirilen'). CSV'ler başka klasördeyse tam yolunu verin.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Cache'i yok say")
    parser.add_argument("--out",      default="besiktas_trafik_lstm.parquet")
    parser.add_argument("--haric",    default=None,
                        help="Dışarıda bırakılacak 6 karakterli hücreler, virgülle ayrılmış. "
                             "Varsayılan: Asya yakası listesi. Hiçbirini çıkarmamak için: --haric \"\"")
    parser.add_argument("--freeflow-q",    type=float, default=0.95,
                        help="Serbest akış hızı için hız yüzdeliği (varsayılan 0.95)")
    parser.add_argument("--freeflow-oran", type=float, default=0.70,
                        help="Serbest akış hızı verinin zamansal ilk kaçta kaçından "
                             "hesaplansın (varsayılan 0.70; 1.0 = tüm veri)")
    parser.add_argument("--freeflow-min-arac", type=int, default=0,
                        help="Serbest akış hızı sadece en az bu kadar araçlı saatlerden "
                             "hesaplansın (varsayılan 0 = filtre yok)")
    args = parser.parse_args()
    haric_hucreler = (HARIC_HUCRELER if args.haric is None else
                      tuple(x.strip() for x in args.haric.split(",") if x.strip()))

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    n_csv = len(list(cache_dir.glob("traffic_density_*.csv")))
    print(f"Cache klasörü: {cache_dir.resolve()}  ({n_csv} CSV)")
    if n_csv == 0 and not args.no_cache:
        print("  [UYARI] Bu klasörde hiç CSV yok, aylar İBB'den İNDİRİLECEK (yavaş ve "
              "zaman aşımına uğrayabilir).")
        print("          CSV'ler başka bir klasördeyse durdurun (Ctrl+C) ve "
              "--cache-dir ile o klasörü gösterin.\n")

    if args.no_cache:
        for f in cache_dir.glob("*.csv"):
            f.unlink()

    if args.quick:
        files = QUICK_FILES
        print(f"[Hızlı mod] 2023-2024: {len(files)} ay")
    elif args.sample:
        files = [MONTHLY_FILES[-1]]
        print(f"[Örnek mod] Tek ay: {files[0][0]}")
    else:
        files = MONTHLY_FILES
        print(f"[Tam mod] {len(files)} ay, Ocak 2020 → Ocak 2025")

    print(f"Çıktı: {args.out}\n")

    all_frames = []
    ham_satir_toplam = 0
    for i, (ym, resource_id) in enumerate(files, 1):
        print(f"[{i:>2}/{len(files)}] {ym} indiriliyor...", end=" ")
        df_raw = download_month(ym, resource_id, cache_dir)

        if df_raw is None:
            print("ATLANDI")
            continue

        ham_satir_toplam += len(df_raw)
        print(f"{len(df_raw):>8,} satır ham →", end=" ")
        df_b = process_month(df_raw, ym)

        if df_b.empty:
            print("Beşiktaş kaydı YOK")
            print(f"  Sütunlar: {list(df_raw.columns[:8])}")
        else:
            print(f"{len(df_b):>6,} Beşiktaş kaydı ✓")
            all_frames.append(df_b)

        time.sleep(0.3)

    if not all_frames:
        print("\n[SONUÇ] Hiç Beşiktaş kaydı bulunamadı.")
        return

    print(f"\n[Birleştirme] {len(all_frames)} ay birleştiriliyor...")
    merged = pd.concat(all_frames, ignore_index=True)

    # Asya yakası hücrelerini çıkar (temizlik ve lag hesabından ÖNCE)
    haric_satir = 0
    if haric_hucreler and "geohash" in merged.columns:
        once = len(merged)
        merged = merged[~merged["geohash"].astype(str).str[:6].isin(haric_hucreler)]
        merged = merged.reset_index(drop=True)
        haric_satir = once - len(merged)
        print(f"  Asya yakası hücreleri çıkarıldı: {haric_satir:,} satır "
              f"({len(haric_hucreler)} hücre listede)")

    if "datetime" in merged.columns:
        merged = merged.sort_values("datetime")

    # Temizlik: hedef artık HIZ (eksik hız doldurulur, 3+ ardışık eksik saat atılır)
    if "speed" not in merged.columns or not merged["speed"].notna().any():
        print("\n[HATA] Veride hız sütunu (AVERAGE_SPEED) bulunamadı, congestion hesaplanamaz.")
        return
    print("\n[Temizlik] Eksik hız verisi dolduruluyor (hedef: speed)...")
    merged, temizlik_sayac = clean_and_fill(merged, target_col="speed")

    # Hıza bağlı congestion_ratio
    print("\n[Congestion] congestion_ratio hızdan hesaplanıyor...")
    merged, congestion_bilgi = add_congestion_from_speed(
        merged, q=args.freeflow_q, ratio=args.freeflow_oran,
        min_arac=args.freeflow_min_arac)

    # LSTM feature'ları
    print("\n[Feature mühendisliği] Lag ve döngüsel özellikler ekleniyor...")
    merged = add_lstm_features(merged)

    # Kaydet
    merged.to_parquet(args.out, index=False, engine="pyarrow")

    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  Toplam satır  : {len(merged):,}")
    print(f"  Tarih aralığı : {merged['date_hour'].min()[:10]} → {merged['date_hour'].max()[:10]}")
    print(f"  Sütunlar      : {list(merged.columns)}")
    print(f"  Çıktı         : {args.out}")
    print(f"{'='*60}")

    print("\nİlk 5 satır:")
    show_cols = ["datetime", "date_hour", "geohash", "lat", "lon",
                 "speed", "freeflow_speed", "congestion_ratio", "density", "is_weekend", "target"]
    show_cols = [c for c in show_cols if c in merged.columns]
    print(merged[show_cols].head().to_string(index=False))

    merged.head(1000).to_csv(
        args.out.replace(".parquet", "_ornek.csv"),
        index=False, encoding="utf-8-sig")
    print(f"  Örnek CSV: {args.out.replace('.parquet', '_ornek.csv')}")

    # Kalite raporu
    kalite = {
        "kaynak":                  "IBB Acik Veri",
        "indirilen_ay_sayisi":     len(all_frames),
        "ham_satir_toplam":        int(ham_satir_toplam),
        "dikdortgen_icindeki_satir": int(len(merged) + haric_satir),
        "haric_hucreler":          list(haric_hucreler),
        "haric_satir":             int(haric_satir),
        "besiktas_filtre_sonrasi": int(len(merged)),
        "filtre_orani_pct":        round(len(merged) / max(ham_satir_toplam, 1) * 100, 2),
        "temizlik":                temizlik_sayac,
        "congestion":              congestion_bilgi,
        "final_satir":             int(len(merged)),
        "final_sutun":             int(len(merged.columns)),
    }
    with open("trafik_kalite.json", "w", encoding="utf-8") as f:
        json.dump(kalite, f, ensure_ascii=False, indent=2)
    print(f"\n  Kalite raporu: trafik_kalite.json")


if __name__ == "__main__":
    main()
