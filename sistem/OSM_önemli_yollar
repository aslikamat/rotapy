"""
OSM → TomTom Sorgu Noktası Seçici
==================================
Her İBB hücresi (6 karakterli geohash, ~600 m x 900 m) için OpenStreetMap'ten
ana yolları (motorway/trunk/primary/secondary/tertiary) bulur ve hücre başına
en önemli N yoldan birer TomTom sorgu noktası seçer.

Neden: İBB bir hücre için o hücredeki yolların toplanmış değerini veriyor,
TomTom ise noktaya en yakın TEK yolu. Hücre başına birkaç ana yoldan
örnek alıp ortalamasını almak iki veriyi birbirine yaklaştırır.

Kurulum:
    pip install requests pandas pyarrow

Kullanım:
    python osm_nokta_sec.py                          # hücre başına 2 yol
    python osm_nokta_sec.py --per-cell 3
    python osm_nokta_sec.py --max-points 100         # kota için üst sınır
    python osm_nokta_sec.py --minor-fallback         # ana yolu olmayan hücrelere sokak ekle
    python osm_nokta_sec.py --refresh                # OSM'den yeniden indir

Çıktı:
    tomtom_noktalar.json   → TomTom script'i bunu okuyacak (ROAD_SEGMENTS formatı)
    tomtom_noktalar.csv    → incelemek / haritada görmek için
    osm_ham.json           → OSM ham cevabı (önbellek; tekrar indirilmez)

Veri kaynağı: © OpenStreetMap contributors (ODbL). Rapor ve sunumda belirtin.
"""

import argparse
import json
import math
import time
from pathlib import Path

import requests

# =============================================================================
# GEOHASH
# =============================================================================

_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def encode_geohash(lat: float, lon: float, precision: int = 7) -> str:
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
            out.append(_B32[ch])
            bit = ch = 0
    return "".join(out)


def decode_bbox(gh: str) -> tuple:
    """Geohash hücresinin sınırları: (güney, kuzey, batı, doğu)."""
    lat_r, lon_r = [-90.0, 90.0], [-180.0, 180.0]
    even = True
    for c in gh:
        cd = _B32.index(c)
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_r[0] + lon_r[1]) / 2
                if cd & mask:
                    lon_r[0] = mid
                else:
                    lon_r[1] = mid
            else:
                mid = (lat_r[0] + lat_r[1]) / 2
                if cd & mask:
                    lat_r[0] = mid
                else:
                    lat_r[1] = mid
            even = not even
    return lat_r[0], lat_r[1], lon_r[0], lon_r[1]


# =============================================================================
# HÜCRELER (İBB verisinden)
# =============================================================================

# İBB verisi yoksa kullanılacak Beşiktaş kutusu: (güney, kuzey, batı, doğu)
BESIKTAS_BBOX = (41.025, 41.075, 28.980, 29.040)


def load_cells(parquet_path: str) -> tuple:
    """İBB parquet'indeki benzersiz 6 karakterli hücreleri döndürür."""
    if parquet_path and Path(parquet_path).exists():
        import pandas as pd
        gh = pd.read_parquet(parquet_path, columns=["geohash"])["geohash"]
        cells = sorted({g[:6] for g in gh.dropna().unique() if len(g) >= 6})
        return cells, f"İBB verisi ({parquet_path})"

    # Yedek: Beşiktaş kutusundaki tüm 6 karakterli hücreler
    s, n, w, e = BESIKTAS_BBOX
    dlat, dlon = 180 / 2 ** 15, 360 / 2 ** 15
    lats = [s + i * dlat / 2 for i in range(int((n - s) / (dlat / 2)) + 1)] + [n]
    lons = [w + i * dlon / 2 for i in range(int((e - w) / (dlon / 2)) + 1)] + [e]
    cells = {encode_geohash(la, lo, 6) for la in lats for lo in lons}
    return sorted(cells), "Beşiktaş kutusu (İBB dosyası bulunamadı)"


# =============================================================================
# OVERPASS (OpenStreetMap)
# =============================================================================

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",   # eski adı overpass.kumi.systems
]

MAIN_CLASSES  = "motorway|trunk|primary|secondary|tertiary"
MINOR_CLASSES = "unclassified|residential|living_street"

CLASS_RANK = {
    "motorway": 0, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4,
    "unclassified": 5, "residential": 6, "living_street": 7,
}
MAIN_MAX_RANK = 4


def build_query(bbox: tuple, include_minor: bool) -> str:
    s, n, w, e = bbox
    classes = MAIN_CLASSES + ("|" + MINOR_CLASSES if include_minor else "")
    return (f'[out:json][timeout:90];'
            f'way["highway"~"^({classes})$"]({s:.5f},{w:.5f},{n:.5f},{e:.5f});'
            f'out geom;')


def fetch_overpass(query: str, retries: int = 3) -> dict:
    last = "bilinmiyor"
    for attempt in range(retries):
        for url in OVERPASS_URLS:
            try:
                r = requests.post(
                    url, data={"data": query}, timeout=120,
                    headers={"User-Agent": "besiktas-rota-projesi/1.0 (TUBITAK 2209-A)"},
                )
                if r.status_code == 200:
                    return r.json()
                last = f"{url} → HTTP {r.status_code}"
            except requests.RequestException as ex:
                last = f"{url} → {ex}"
        wait = 10 * (attempt + 1)
        print(f"  [UYARI] Overpass cevap vermedi ({last}), {wait} sn sonra tekrar...")
        time.sleep(wait)
    raise RuntimeError(f"Overpass'a ulaşılamadı: {last}")


# =============================================================================
# GEOMETRİ
# =============================================================================

def meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Kısa mesafeler için yaklaşık metre hesabı."""
    mlat = 111_320.0
    mlon = 111_320.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot((lat2 - lat1) * mlat, (lon2 - lon1) * mlon)


def densify(coords: list, step: float = 10.0) -> list:
    """
    Yolu ~step metre aralıklı noktalara böler.
    Dönen liste: (lat, lon, ağırlık_m) — ağırlık o noktanın temsil ettiği yol uzunluğu.
    Hücre sınırında kırpmayı basitleştirir: hücre içinde kalan noktaları sayarız.
    """
    out = []
    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        d = meters(a[0], a[1], b[0], b[1])
        k = max(1, int(d // step))
        for j in range(k):
            t = j / k
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, d / k))
    out.append((coords[-1][0], coords[-1][1], 0.0))
    return out


def middle_point(inside: list) -> tuple:
    """Hücre içindeki yol parçasının uzunluk olarak ortasındaki nokta."""
    total = sum(p[2] for p in inside)
    acc = 0.0
    for lat, lon, wgt in inside:
        acc += wgt
        if acc >= total / 2:
            return lat, lon
    return inside[-1][0], inside[-1][1]


# =============================================================================
# SEÇİM
# =============================================================================

def groups_for_cell(cell_bbox: tuple, ways: list, step: float = 10.0) -> dict:
    """
    Hücreyle kesişen yolları isme göre gruplar (OSM bir caddeyi çok parçaya
    böler, çift yönlü caddeler iki paralel çizgi olur — isimle birleştiriyoruz).
    """
    s, n, w, e = cell_bbox
    groups = {}
    for el in ways:
        tags = el.get("tags", {})
        hw = tags.get("highway")
        if hw not in CLASS_RANK:
            continue
        coords = [(g["lat"], g["lon"]) for g in el.get("geometry", [])]
        if len(coords) < 2:
            continue
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        if max(lats) < s or min(lats) > n or max(lons) < w or min(lons) > e:
            continue  # hızlı eleme: yolun kutusu hücreyle kesişmiyor

        inside = [p for p in densify(coords, step)
                  if s <= p[0] < n and w <= p[1] < e]
        length = sum(p[2] for p in inside)
        if length <= 0:
            continue

        name = tags.get("name") or tags.get("ref") or f"way{el.get('id')}"
        g = groups.setdefault(name, {
            "road": name, "rank": 99, "highway": hw,
            "length": 0.0, "best_len": 0.0, "best_pts": None,
        })
        g["length"] += length
        if CLASS_RANK[hw] < g["rank"]:
            g["rank"], g["highway"] = CLASS_RANK[hw], hw
        if length > g["best_len"]:
            g["best_len"], g["best_pts"] = length, inside
    return groups


def select_for_cell(groups: dict, per_cell: int, min_len: float,
                    min_sep: float, minor_fallback: bool) -> list:
    """
    Önce ana yollar (rank <= tertiary), önem sırasına ve hücre içi uzunluğa göre.
    Hücrede hiç ana yol yoksa ve minor_fallback açıksa sokaklardan seçer.
    """
    pool = [g for g in groups.values()
            if g["length"] >= min_len and g["rank"] <= MAIN_MAX_RANK]
    if not pool and minor_fallback:
        pool = [g for g in groups.values() if g["length"] >= min_len]
    pool.sort(key=lambda g: (g["rank"], -g["length"]))

    chosen = []
    for g in pool:
        lat, lon = middle_point(g["best_pts"])
        # İki nokta çok yakınsa TomTom ikisini de aynı yola oturtur — atla
        if any(meters(lat, lon, c["lat"], c["lon"]) < min_sep for c in chosen):
            continue
        chosen.append({
            "road": g["road"], "highway": g["highway"],
            "lat": round(lat, 6), "lon": round(lon, 6),
            "length_m": int(round(min(g["length"], 500))),
            "_rank": g["rank"], "_len": g["length"],
        })
        if len(chosen) == per_cell:
            break
    return chosen


# =============================================================================
# ANA AKIŞ
# =============================================================================

def get_osm(cache_path: Path, bbox: tuple, include_minor: bool, refresh: bool) -> dict:
    """Önbellek geçerliyse onu kullanır (aynı bölge ve gereken yol sınıfları)."""
    s, n, w, e = bbox
    if cache_path.exists() and not refresh:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        meta = data.get("_meta", {})
        mb = meta.get("bbox", [1e9, -1e9, 1e9, -1e9])
        covers = mb[0] <= s and mb[1] >= n and mb[2] <= w and mb[3] >= e
        if covers and (meta.get("minor") or not include_minor):
            print(f"  Önbellek kullanılıyor: {cache_path}  (yenilemek için --refresh)")
            return data
        print("  Önbellek bu istek için yetersiz, OSM'den yeniden indiriliyor...")

    print("  OSM (Overpass) sorgulanıyor, 10-60 sn sürebilir...")
    data = fetch_overpass(build_query(bbox, include_minor))
    data["_meta"] = {"bbox": [s, n, w, e], "minor": include_minor}
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="OSM → TomTom sorgu noktası seçici")
    ap.add_argument("--ibb", default="besiktas_trafik_lstm.parquet",
                    help="İBB parquet'i (hücre listesi buradan alınır)")
    ap.add_argument("--per-cell", type=int, default=2,
                    help="Hücre başına en fazla kaç yol (varsayılan 2)")
    ap.add_argument("--min-len", type=float, default=80,
                    help="Hücre içinde en az bu kadar metre yol olmalı (varsayılan 80)")
    ap.add_argument("--min-sep", type=float, default=100,
                    help="Aynı hücredeki iki nokta arası en az metre (varsayılan 100)")
    ap.add_argument("--max-points", type=int, default=None,
                    help="Toplam nokta üst sınırı (API kotası için)")
    ap.add_argument("--minor-fallback", action="store_true",
                    help="Hiç ana yolu olmayan hücrelerde sokaklardan seç")
    ap.add_argument("--cache", default="osm_ham.json")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default="tomtom_noktalar",
                    help="Çıktı dosya öneki (.json ve .csv üretilir)")
    args = ap.parse_args()

    print("\n[1/3] İBB hücreleri belirleniyor...")
    cells, kaynak = load_cells(args.ibb)
    print(f"  Kaynak: {kaynak}")
    print(f"  {len(cells)} adet 6 karakterli hücre")

    bboxes = {c: decode_bbox(c) for c in cells}
    bbox = (min(b[0] for b in bboxes.values()), max(b[1] for b in bboxes.values()),
            min(b[2] for b in bboxes.values()), max(b[3] for b in bboxes.values()))

    print("\n[2/3] OpenStreetMap yolları alınıyor...")
    data = get_osm(Path(args.cache), bbox, args.minor_fallback, args.refresh)
    ways = [el for el in data.get("elements", []) if el.get("type") == "way"]
    print(f"  {len(ways):,} yol parçası")

    print("\n[3/3] Her hücre için nokta seçiliyor...")
    points, bos_hucre = [], []
    for cell in cells:
        groups = groups_for_cell(bboxes[cell], ways)
        chosen = select_for_cell(groups, args.per_cell, args.min_len,
                                 args.min_sep, args.minor_fallback)
        if not chosen:
            bos_hucre.append(cell)
            continue
        for k, c in enumerate(chosen, 1):
            points.append({
                "id": f"{cell}-{k}", "road": c["road"],
                "lat": c["lat"], "lon": c["lon"], "length_m": c["length_m"],
                "cell": cell, "geohash": encode_geohash(c["lat"], c["lon"], 7),
                "highway": c["highway"],
                "_pick": k, "_rank": c["_rank"], "_len": c["_len"],
            })

    if args.max_points and len(points) > args.max_points:
        points.sort(key=lambda p: (p["_pick"], p["_rank"], -p["_len"]))
        cikarilan = len(points) - args.max_points
        points = points[:args.max_points]
        print(f"  [BİLGİ] Kota sınırı için {cikarilan} nokta çıkarıldı "
              f"(ikinci/üçüncü seçimlerden, önemsiz yollar önce)")
    points.sort(key=lambda p: (p["cell"], p["_pick"]))

    temiz = [{k: v for k, v in p.items() if not k.startswith("_")} for p in points]
    Path(args.out + ".json").write_text(
        json.dumps(temiz, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["id", "cell", "geohash", "road", "highway", "lat", "lon", "length_m"]
    with open(args.out + ".csv", "w", encoding="utf-8-sig") as f:
        f.write(",".join(cols) + "\n")
        for p in temiz:
            f.write(",".join('"{}"'.format(str(p[c]).replace('"', "'")) for c in cols) + "\n")

    # ── Özet ────────────────────────────────────────────────────────────────
    hucre_sayisi = len({p["cell"] for p in temiz})
    ayni_g7 = len(temiz) - len({p["geohash"] for p in temiz})
    sinif = {}
    for p in temiz:
        sinif[p["highway"]] = sinif.get(p["highway"], 0) + 1

    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  Nokta seçilen hücre : {hucre_sayisi} / {len(cells)}")
    print(f"  Toplam nokta        : {len(temiz)}")
    print(f"  Yol sınıfı dağılımı : {sinif}")
    if bos_hucre:
        print(f"  Nokta bulunamayan   : {len(bos_hucre)} hücre → {', '.join(bos_hucre)}")
        print(f"     (deniz/orman olabilir; sokak eklemek için --minor-fallback)")
    if ayni_g7:
        print(f"  [UYARI] {ayni_g7} nokta başka bir noktayla aynı 7 karakterli geohash'te")
    print(f"\n  Tahmini TomTom isteği:")
    for interval in (60, 10):
        print(f"    {interval:>2} dk aralıkla: {len(temiz) * (1440 // interval):>7,} istek/gün")
    print(f"\n  Çıktı: {args.out}.json  ve  {args.out}.csv")
    print(f"  Haritada görmek için CSV'yi Google My Maps'e yükleyebilirsiniz.")
    print(f"  © OpenStreetMap contributors")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
