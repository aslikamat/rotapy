"""
İBB ham dosyalarından Beşiktaş kutusundaki hücreleri (geohash) listeler.

Veri Ocak 2020 → Ocak 2025 arası (61 ay). Varsayılan olarak hepsini okur.

Kullanım:
    python kontrol.py                                   # 2020-01 → 2025-01, tüm aylar
    python kontrol.py "C:\\rotapy\\eğitim\\indirilen\\traffic_density_202[34]*.csv"   # sadece 2023-2024
    python kontrol.py "C:\\rotapy\\eğitim\\indirilen\\traffic_density_202001.csv"      # tek ay

Çıktı:
    - Ekranda: her ayın hücre sayısı, toplam hücre listesi
    - ibb_hucreler.csv : hücre, 7 karakterli geohash, koordinat, kaç ayda göründü
"""

import glob
import re
import sys

import pandas as pd

VARSAYILAN = r"C:\rotapy\eğitim\indirilen\traffic_density_20*.csv"
BBOX = dict(lat_min=41.025, lat_max=41.075, lon_min=28.980, lon_max=29.040)
_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash(lat: float, lon: float, precision: int) -> str:
    lat_r, lon_r = [-90.0, 90.0], [-180.0, 180.0]
    out, bit, ch, even = [], 0, 0, True
    bits = [16, 8, 4, 2, 1]
    while len(out) < precision:
        if even:
            mid = (lon_r[0] + lon_r[1]) / 2
            if lon > mid:
                ch |= bits[bit]; lon_r[0] = mid
            else:
                lon_r[1] = mid
        else:
            mid = (lat_r[0] + lat_r[1]) / 2
            if lat > mid:
                ch |= bits[bit]; lat_r[0] = mid
            else:
                lat_r[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            out.append(_B32[ch]); bit = ch = 0
    return "".join(out)


def ay_listesi(bas: str, son: str) -> list:
    """'202001' → '202501' arası tüm ayları üretir."""
    y, m = int(bas[:4]), int(bas[4:])
    out = []
    while f"{y}{m:02d}" <= son:
        out.append(f"{y}{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def sutun(d: pd.DataFrame, *adaylar: str):
    kucuk = {c.lower(): c for c in d.columns}
    for a in adaylar:
        if a in kucuk:
            return kucuk[a]
    return None


def main() -> None:
    desen = sys.argv[1] if len(sys.argv) > 1 else VARSAYILAN
    dosyalar = sorted(glob.glob(desen))
    if not dosyalar:
        print("Dosya bulunamadı:", desen)
        return
    print(f"{len(dosyalar)} dosya okunacak\n")

    gorulen = {}          # (lat, lon) -> görüldüğü aylar
    ham_geohash = set()   # ham dosyada geohash sütunu varsa değerleri
    ilk = True

    for f in dosyalar:
        ay = (re.search(r"(\d{6})", f.split("\\")[-1].split("/")[-1]) or [None, f])[1]
        d = pd.read_csv(
            f, low_memory=False,
            usecols=lambda c: c.upper() in ("LATITUDE", "LONGITUDE", "LAT", "LON", "GEOHASH"),
        )
        if ilk:
            print("Okunan sütunlar:", list(d.columns), "\n")
            ilk = False
        lat = sutun(d, "latitude", "lat")
        lon = sutun(d, "longitude", "lon")
        d[lat] = pd.to_numeric(d[lat], errors="coerce")
        d[lon] = pd.to_numeric(d[lon], errors="coerce")
        b = d[d[lat].between(BBOX["lat_min"], BBOX["lat_max"]) &
              d[lon].between(BBOX["lon_min"], BBOX["lon_max"])]
        noktalar = set(zip(b[lat].round(6), b[lon].round(6)))
        for p in noktalar:
            gorulen.setdefault(p, set()).add(ay)
        g = sutun(d, "geohash")
        if g:
            ham_geohash |= set(b[g].dropna().astype(str).unique())
        print(f"{ay}: kutu içinde {len(noktalar)} hücre")

    okunan = sorted({a for aylar in gorulen.values() for a in aylar} |
                    {re.search(r"(\d{6})", f.split("\\")[-1].split("/")[-1]).group(1)
                     for f in dosyalar if re.search(r"(\d{6})", f)})
    if okunan and all(len(a) == 6 and a.isdigit() for a in okunan):
        beklenen = ay_listesi(okunan[0], okunan[-1])
        eksik = [a for a in beklenen if a not in okunan]
        print(f"\nKapsanan dönem: {okunan[0]} → {okunan[-1]} ({len(okunan)} dosya, "
              f"{len(beklenen)} aylık aralık)")
        print("Eksik ay:", ", ".join(eksik) if eksik else "yok")
        yillar = sorted({a[:4] for a in okunan})
        print("Yıllara göre benzersiz hücre:")
        for y in yillar:
            n = sum(1 for aylar in gorulen.values() if any(a.startswith(y) for a in aylar))
            print(f"  {y}: {n} hücre")

    satirlar = []
    for (la, lo), aylar in gorulen.items():
        satirlar.append({
            "cell6": geohash(la, lo, 6), "geohash7": geohash(la, lo, 7),
            "lat": la, "lon": lo, "ay_sayisi": len(aylar),
        })
    tablo = pd.DataFrame(satirlar).sort_values(["cell6", "geohash7"]).reset_index(drop=True)
    tablo.to_csv("ibb_hucreler.csv", index=False, encoding="utf-8-sig")

    print(f"\nToplam benzersiz hücre (tüm aylar birleşimi): {len(tablo)}")
    print(f"Benzersiz 6 karakterli kod: {tablo['cell6'].nunique()}")
    if ham_geohash:
        uz = sorted({len(x) for x in ham_geohash})
        print(f"Ham dosyadaki geohash sütunu: {len(ham_geohash)} değer, uzunluk {uz}")
    print(f"\nTüm {len(dosyalar)} ayda görünen hücre: {(tablo['ay_sayisi'] == len(dosyalar)).sum()}")
    print("Sadece bazı aylarda görünenler:")
    kismi = tablo[tablo["ay_sayisi"] < len(dosyalar)]
    print(kismi.to_string(index=False) if len(kismi) else "  (yok, hepsi her ayda var)")

    print("\nHÜCRE LİSTESİ")
    print(tablo.to_string(index=False))
    print("\nKaydedildi: ibb_hucreler.csv")


if __name__ == "__main__":
    main()
