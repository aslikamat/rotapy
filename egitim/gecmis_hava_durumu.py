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
    gecmis_hava_durumu_ornek.csv → ilk 1000 satır örnek CSV

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
    Aynı sütun isimleri kullanılıyor. Noktalar trafik verisindeki 35
    İBB hücresine göre güncellendi; sistem/anlik_hava_durumu.py hâlâ eski
    58 noktayı kullanıyor (henüz güncellenmedi).

Veri hacmi: 5 yıl × 35 nokta × saatlik ≈ 1.5 milyon satır.
Gerekirse --start/--end ile aralığı daraltın.
"""

import requests
import pandas as pd
import argparse
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# BEŞİKTAŞ NOKTALARI
# Trafik verisindeki (besiktas_trafik_lstm.parquet) 35 İBB hücresi.
# Koordinatlar her 7 karakterli geohash hücresinin merkezi — encode_geohash
# trafikteki geohash'in AYNISINI üretir, böylece birlestirme.py'de
# geohash + date_hour join'i birebir eşleşir.
# =============================================================================

POINTS = [
    {"lat": 41.02638, "lon": 28.98674},  # sxk97se
    {"lat": 41.03188, "lon": 28.98674},  # sxk97te
    {"lat": 41.02638, "lon": 28.99773},  # sxk97ue
    {"lat": 41.03188, "lon": 28.99773},  # sxk97ve
    {"lat": 41.03737, "lon": 28.98674},  # sxk97we
    {"lat": 41.04286, "lon": 28.98674},  # sxk97xe
    {"lat": 41.03737, "lon": 28.99773},  # sxk97ye
    {"lat": 41.04286, "lon": 28.99773},  # sxk97ze
    {"lat": 41.04836, "lon": 28.98674},  # sxk9e8e
    {"lat": 41.05385, "lon": 28.98674},  # sxk9e9e
    {"lat": 41.04836, "lon": 28.99773},  # sxk9ebe
    {"lat": 41.05385, "lon": 28.99773},  # sxk9ece
    {"lat": 41.05934, "lon": 28.98674},  # sxk9ede
    {"lat": 41.06483, "lon": 28.98674},  # sxk9eee
    {"lat": 41.05934, "lon": 28.99773},  # sxk9efe
    {"lat": 41.06483, "lon": 28.99773},  # sxk9ege
    {"lat": 41.07033, "lon": 28.98674},  # sxk9ese
    {"lat": 41.07033, "lon": 28.99773},  # sxk9eue
    {"lat": 41.04286, "lon": 29.00871},  # sxk9kpe
    {"lat": 41.04286, "lon": 29.0197},  # sxk9kre
    {"lat": 41.04836, "lon": 29.00871},  # sxk9s0e
    {"lat": 41.05385, "lon": 29.00871},  # sxk9s1e
    {"lat": 41.04836, "lon": 29.0197},  # sxk9s2e
    {"lat": 41.05385, "lon": 29.0197},  # sxk9s3e
    {"lat": 41.05934, "lon": 29.00871},  # sxk9s4e
    {"lat": 41.06483, "lon": 29.00871},  # sxk9s5e
    {"lat": 41.05934, "lon": 29.0197},  # sxk9s6e
    {"lat": 41.06483, "lon": 29.0197},  # sxk9s7e
    {"lat": 41.04836, "lon": 29.03069},  # sxk9s8e
    {"lat": 41.05385, "lon": 29.03069},  # sxk9s9e
    {"lat": 41.05934, "lon": 29.03069},  # sxk9sde
    {"lat": 41.06483, "lon": 29.03069},  # sxk9see
    {"lat": 41.07033, "lon": 29.00871},  # sxk9she
    {"lat": 41.07033, "lon": 29.0197},  # sxk9ske
    {"lat": 41.07033, "lon": 29.03069},  # sxk9sse
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
    45-48 : Sis                  → yavaşlama (arşivde gelmez → sis_tespit)
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


def sis_tespit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Open-Meteo arşivi (ERA5) sis kodu (45-48) üretmiyor; bu yüzden is_foggy
    hep 0 kalıyordu. Sisi meteorolojik koşullardan türetiyoruz:
      nem >= %97 (çiy noktası ≈ sıcaklık) + rüzgâr < 10 km/s + yağış yok
    WMO kodu 45-48 gelirse o da sis sayılır. is_bad_weather da güncellenir.
    """
    turetilen = (
        (df["humidity_pct"] >= 97)
        & (df["wind_speed_kmh"] < 10)
        & (df["precipitation_mm"] == 0)
    )
    df["is_foggy"]       = (df["is_foggy"].astype(bool) | turetilen).astype(int)
    df["is_bad_weather"] = (df["is_bad_weather"].astype(bool) | df["is_foggy"].astype(bool)).astype(int)
    return df


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
    """Tüm noktalar için tek API isteği parametresi oluşturur."""
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


# Open-Meteo ücretsiz limitleri: dakikalık / saatlik / günlük "ağırlıklı" çağrı.
# Tüm noktalar × 1 yıl tek istek dakikalık limitin büyük kısmını yiyor; bu yüzden
# parçalar arası bekliyoruz ve 429 gelirse tekrar deniyoruz.
CHUNK_BEKLEME_SN = 65
MAX_DENEME       = 6


def _get_with_retry(params: dict) -> pd.DataFrame:
    """
    429 (limit aşımı) ve geçici ağ hatalarında bekleyip tekrar dener.
    Tüm denemeler başarısız olursa HATA FIRLATIR — eksik yıllarla sessizce
    kaydetmek yerine (eskiden yalnızca 2020 kaydediliyordu) durmak daha güvenli.
    """
    for deneme in range(1, MAX_DENEME + 1):
        try:
            r = requests.get(HISTORICAL_URL, params=params, timeout=120)
        except requests.RequestException as e:
            bekle = 30 * deneme
            print(f"\n    [Ağ hatası] {e} → {bekle} sn sonra tekrar ({deneme}/{MAX_DENEME})", end=" ")
            time.sleep(bekle)
            continue

        if r.status_code == 429:
            reason = r.json().get("reason", "") if r.headers.get("content-type", "").startswith("application/json") else r.text
            if "Daily" in reason:
                raise RuntimeError(f"Open-Meteo günlük limit aşıldı, yarın tekrar deneyin: {reason}")
            bekle = 3600 if "Hourly" in reason else 65
            print(f"\n    [429] {reason} → {bekle} sn bekleniyor ({deneme}/{MAX_DENEME})", end=" ")
            time.sleep(bekle)
            continue

        r.raise_for_status()
        return _parse_multi_location_response(r.json())

    raise RuntimeError(f"{params.get('start_date')} → {params.get('end_date')} parçası "
                       f"{MAX_DENEME} denemede çekilemedi.")


def _chunk_listesi(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Tarih aralığını yıllık parçalara böler."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    chunks  = []
    current = start
    while current <= end:
        chunk_end = min(datetime(current.year, 12, 31), end)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = datetime(current.year + 1, 1, 1)
    return chunks


def _beklenen_satir(chunk_start: str, chunk_end: str) -> int:
    """Parça tamsa olması gereken satır sayısı: saat sayısı × nokta sayısı."""
    gun = (datetime.strptime(chunk_end, "%Y-%m-%d")
           - datetime.strptime(chunk_start, "%Y-%m-%d")).days + 1
    return gun * 24 * len(POINTS)


def _parca_yolu(parca_dir: Path, chunk_start: str, chunk_end: str) -> Path:
    return parca_dir / f"hava_{chunk_start}_{chunk_end}.parquet"


def _parca_tam_mi(path: Path, chunk_start: str, chunk_end: str) -> bool:
    if not path.exists():
        return False
    try:
        n = len(pd.read_parquet(path, columns=["timestamp"]))
    except Exception:
        return False
    return n == _beklenen_satir(chunk_start, chunk_end)


def _mevcut_ciktidan_tohumla(out_path: str | None, chunks, parca_dir: Path) -> None:
    """
    Parça dosyası olmayan ama ana çıktı dosyasında tam olarak bulunan yılları
    parça klasörüne çıkarır — böylece daha önce çekilmiş yıllar için
    (ör. eski 2020 verisi) API'ye tekrar istek atılmaz.
    """
    if not out_path or not Path(out_path).exists():
        return
    eksikler = [(s, e) for s, e in chunks
                if not _parca_tam_mi(_parca_yolu(parca_dir, s, e), s, e)]
    if not eksikler:
        return
    try:
        mevcut = pd.read_parquet(out_path)
    except Exception:
        return
    ts = pd.to_datetime(mevcut["timestamp"])
    for s, e in eksikler:
        maske = (ts >= pd.Timestamp(s)) & (ts < pd.Timestamp(e) + pd.Timedelta(days=1))
        parca = mevcut[maske]
        if len(parca) == _beklenen_satir(s, e):
            parca_dir.mkdir(parents=True, exist_ok=True)
            parca.to_parquet(_parca_yolu(parca_dir, s, e), index=False, engine="pyarrow")
            print(f"  [Tohum] {s} → {e} mevcut {out_path} dosyasından alındı ({len(parca):,} satır)")


def _parcalari_birlestir(chunks, parca_dir: Path) -> pd.DataFrame:
    """Diskteki tam parçaları birleştirir (eksik olanlar atlanır)."""
    dfs = [pd.read_parquet(_parca_yolu(parca_dir, s, e))
           for s, e in chunks if _parca_tam_mi(_parca_yolu(parca_dir, s, e), s, e)]
    if not dfs:
        return pd.DataFrame()
    return (pd.concat(dfs, ignore_index=True)
              .sort_values(["timestamp", "geohash"])
              .reset_index(drop=True))


def fetch_historical(
    start_date: str = "2020-01-01",
    end_date:   str = "2025-01-31",
    parca_dir:  str | Path = "hava_parcalar",
    out_path:   str | None = None,
) -> pd.DataFrame:
    """
    Belirtilen tarih aralığı için tüm noktaların saatlik geçmiş hava verisini
    yıllık parçalar halinde çeker. Her parça çekilir çekilmez
    `parca_dir` içine kaydedilir; daha önce tam kaydedilmiş parçalar için
    API'ye istek atılmaz. Limit aşılıp script yarıda kesilirse, tekrar
    çalıştırıldığında yalnızca eksik yıllar çekilir.
    """
    parca_dir = Path(parca_dir)
    print(f"\n[Geçmiş Hava] {start_date} → {end_date} çekiliyor "
          f"({len(POINTS)} nokta/geohash, parçalar: {parca_dir})...")

    chunks = _chunk_listesi(start_date, end_date)
    _mevcut_ciktidan_tohumla(out_path, chunks, parca_dir)

    istek_atildi = False
    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        yol = _parca_yolu(parca_dir, chunk_start, chunk_end)
        print(f"  [{i}/{len(chunks)}] {chunk_start} → {chunk_end}...", end=" ")
        if _parca_tam_mi(yol, chunk_start, chunk_end):
            print("zaten kayıtlı, atlandı ✓")
            continue

        if istek_atildi:
            time.sleep(CHUNK_BEKLEME_SN)   # dakikalık limite takılmamak için
        params = _multi_location_params({
            "start_date": chunk_start,
            "end_date":   chunk_end,
        })
        df = _process(_get_with_retry(params))
        istek_atildi = True

        parca_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(yol, index=False, engine="pyarrow")
        uyari = "" if len(df) == _beklenen_satir(chunk_start, chunk_end) else \
            f" [UYARI: beklenen {_beklenen_satir(chunk_start, chunk_end):,}]"
        print(f"{len(df):,} satır ✓ kaydedildi{uyari}")

    return _parcalari_birlestir(chunks, parca_dir)


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
    df = sis_tespit(df)

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

    csv_path = path.replace(".parquet", "_ornek.csv")
    df.head(1000).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [Örnek CSV] {csv_path}")

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
        description="Open-Meteo Beşiktaş Geçmiş Hava Verisi (geohash bazlı, 35 trafik hücresi)"
    )
    parser.add_argument("--start", default="2020-01-01",
                        help="Başlangıç tarihi (varsayılan: 2020-01-01)")
    parser.add_argument("--end",   default="2025-01-31",
                        help="Bitiş tarihi (varsayılan: 2025-01-31, trafik verisinin sonu)")
    parser.add_argument("--out",   default="gecmis_hava_durumu.parquet")
    parser.add_argument("--parca-dir", default=None,
                        help="Yıllık parçaların kaydedileceği klasör "
                             "(varsayılan: çıktı dosyasının yanında hava_parcalar/)")
    args = parser.parse_args()
    parca_dir = Path(args.parca_dir) if args.parca_dir else Path(args.out).parent / "hava_parcalar"

    try:
        df = fetch_historical(args.start, args.end, parca_dir=parca_dir, out_path=args.out)
    except RuntimeError as e:
        print(f"\n  [HATA] {e}")
        print("  Çekilen parçalar kaydedildi; tekrar çalıştırınca yalnızca eksikler çekilecek.")
        kismi = _parcalari_birlestir(_chunk_listesi(args.start, args.end), parca_dir)
        if not kismi.empty:
            print("  Şimdiye kadarki parçalar ana dosyaya yazılıyor (kısmi):")
            save(kismi, args.out)
        sys.exit(1)

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
