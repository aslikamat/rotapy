"""
OSM → TomTom Sorgu Noktası Seçici
==================================
Eğitim verisindeki İBB hücreleri (6 karakterli geohash, ~600 m x 900 m) için
OpenStreetMap'ten ana yolları (motorway/trunk/primary/secondary/tertiary) bulur ve
hücre başına en önemli N yoldan birer TomTom sorgu noktası seçer.

Hücre listesi: varsayılan olarak eğitim verisinden okunur
(egitim/besiktas_trafik_lstm.parquet → benzersiz geohash[:6], şu an 35 hücre).
Böylece canlı veri, modelin eğitimde gördüğü hücrelerle birebir aynı olur
(Üsküdar tarafı hücreleri eğitimde hariç tutulduğu için burada da yoktur).
Ayrıca HARIC_HUCRELER (Asya yakası / deniz) her kaynakta ayrıca çıkarılır —
eğitim verisi değişirse bile bu hücreler yanlışlıkla girmesin diye.
OSM'e de SADECE kalan hücrelerin içindeki yollar sorulur.

Neden: İBB bir hücre için o hücredeki yolların toplanmış değerini veriyor,
TomTom ise noktaya en yakın TEK yolu. Hücre başına birkaç ana yoldan
örnek alıp ortalamasını almak iki veriyi birbirine yaklaştırır.

Seçim kuralları:
  - Hücre başına ilk 2 yol: önem sırasına (motorway > trunk > primary >
    secondary > tertiary) ve hücre içi uzunluğa göre.
  - 3. yol: sadece secondary veya üstü VE hücre içinde en az 200 m ise
    (--ucuncu-min-sinif / --ucuncu-min-uzunluk). Değilse hücre 2 noktada kalır.
  - ZORUNLU_YOLLAR: bu kurallardan bağımsız eklenir (en uzun olduğu hücreye),
    --max-points sınırında da korunur.
  - Boş kalan hücre (ör. büyük kısmı deniz): önce hücre içinde daha kısa
    (>= --bos-min-len, varsayılan 20 m) ana yola, sonra --minor-fallback ile
    sokağa bakılır; hücre başına --bos-per-cell (varsayılan 1) nokta seçilir.
    Bu noktalar çıktıda "yedek" sütunuyla işaretlenir.

Kurulum:
    pip install requests

Kullanım:
    python osm_nokta_sec.py                          # eğitim verisindeki hücreler, hücre başına 3 yol, en fazla 100 nokta
    python osm_nokta_sec.py --per-cell 2
    python osm_nokta_sec.py --max-points 0           # nokta sınırını kaldır
    python osm_nokta_sec.py --minor-fallback         # kısa ana yol da yoksa boş hücreye sokak ekle
    python osm_nokta_sec.py --bos-min-len 10         # boş hücrelerde daha kısa yolları da kabul et
    python osm_nokta_sec.py --haric ""               # HARIC_HUCRELER'i uygulama
    python osm_nokta_sec.py --refresh                # OSM'den yeniden indir
    python osm_nokta_sec.py --hucreler ibb_hucreler.csv --min-ay 12   # listeyi CSV'den oku

Çıktı:
    tomtom_noktalar.json   → TomTom script'i bunu okuyacak (ROAD_SEGMENTS formatı)
    tomtom_noktalar.csv    → incelemek / haritada görmek için
    osm_ham.json           → OSM ham cevabı (önbellek; tekrar indirilmez)

Veri kaynağı: © OpenStreetMap contributors (ODbL). Rapor ve sunumda belirtin.
"""

import argparse
import csv
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

# Varsayılan hücre kaynağı: modelin eğitildiği trafik verisi.
# Script nereden çalıştırılırsa çalıştırılsın proje köküne göre bulunur.
EGITIM_TRAFIK = Path(__file__).resolve().parent.parent / "egitim" / "besiktas_trafik_lstm.parquet"

# Dışarıda bırakılan hücreler: Asya yakası (Üsküdar/Kuzguncuk/Beylerbeyi tarafı) ve
# Boğaz'ın ortasında kalan deniz hücresi. Kutunun güneydoğu bloğu + 15 Temmuz Köprüsü'nün
# Asya ayağı. Koordinat ve yol adlarından belirlendi; --haric ile değiştirilebilir.
# egitim/gecmis_trafik_ibb_speed.py'deki HARIC_HUCRELER ile aynı liste.
#   sxk9kh  Üsküdar Harem Sahil Yolu, Doğancılar Cd.
#   sxk9kk  Selmani Pak Cd., Hakimiyeti Milliye Cd.
#   sxk9km  Paşa Limanı Cd., Hüseyin Baykara Sk.
#   sxk9ks  Cumhuriyet Cd., Nuh Kuyusu Cd.
#   sxk9kt  İcadiye Cd., Çifteçınar Sk.
#   sxk9kw  Paşa Limanı Cd., Kuzguncuk Çarşısı Cd.
#   sxk9kx  15 Temmuz Köprüsü Asya ayağı
#   sxk9kq  Boğaz'ın ortası (yol yok)
#   sxk9kj, sxk9kn  sadece 1 ayda görünen nadir hücreler (aynı blokta)
HARIC_HUCRELER = (
    "sxk9kh", "sxk9kj", "sxk9kk", "sxk9km", "sxk9kn",
    "sxk9kq", "sxk9ks", "sxk9kt", "sxk9kw", "sxk9kx",
)


def _hucre_eslemesi(geohashler) -> dict:
    """
    7 karakterli geohash'lerden {hücre6: geohash7} eşlemesi kurar.
    Değerler HESAPLANMAZ, kaynaktaki geohash'in aynısı kullanılır: hücre merkezi
    iki alt hücrenin sınırına denk geldiği için yeniden hesaplamak farklı bir
    7. karakter verebiliyor (ör. sxk9s3e yerine sxk9s37).
    Bir hücrede birden fazla 7 karakterli geohash varsa hangisinin eğitimdeki
    karşılık olduğu belirsizdir → hata verir.
    """
    eslesme = {}
    for g in geohashler:
        if not isinstance(g, str) or len(g) < 7:
            continue
        eslesme.setdefault(g[:6], set()).add(g[:7])
    coklu = {c: sorted(v) for c, v in eslesme.items() if len(v) > 1}
    if coklu:
        raise ValueError(f"Bazı hücrelerde birden fazla 7 karakterli geohash var: {coklu}")
    return {c: next(iter(v)) for c, v in eslesme.items()}


def load_cells(kaynak: str = None, min_ay: int = 12, haric=None) -> tuple:
    """
    Hücre listesi (6 karakterli geohash) ve her hücrenin kaynaktaki 7 karakterli
    geohash'i. Dönüş: (hücreler, {hücre6: geohash7}, kaynak açıklaması)
      kaynak yok      → eğitim verisi (EGITIM_TRAFIK) içindeki tüm benzersiz geohash'ler
      kaynak .csv     → kontrol.py'nin ibb_hucreler.csv'si; ay_sayisi >= min_ay olanlar
                        (geohash7 sütunundan)
      kaynak .parquet → içindeki tüm benzersiz geohash'ler (nadir hücre filtresi YOK)
    haric: dışarıda bırakılacak hücreler (None → HARIC_HUCRELER, [] → hiçbiri).
    Dosya bulunamazsa hata verir, sessizce başka listeye geçmez.
    """
    p = Path(kaynak) if kaynak else EGITIM_TRAFIK
    if not p.exists():
        raise FileNotFoundError(f"Hücre dosyası bulunamadı: {p}")
    if p.suffix.lower() == ".csv":
        with open(p, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        eslesme = _hucre_eslemesi(r["geohash7"] for r in rows if int(r["ay_sayisi"]) >= min_ay)
        etiket = f"{p} (ay_sayisi >= {min_ay})"
    else:
        import pandas as pd
        gh = pd.read_parquet(p, columns=["geohash"])["geohash"]
        eslesme = _hucre_eslemesi(gh.dropna().unique())
        etiket = f"{p} (parquet, nadir hücre filtresi yok)"

    haric = set(HARIC_HUCRELER if haric is None else haric)
    cikan = [c for c in eslesme if c in haric]
    for c in cikan:
        del eslesme[c]
    if cikan:
        etiket += f"; {len(cikan)} hücre hariç (Asya yakası/deniz)"
    return sorted(eslesme), eslesme, etiket


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

# Otomatik seçime girmeyen ama yerel trafik için önemli yollar — her zaman eklenir.
# Ad OSM'deki "name" etiketiyle birebir aynı olmalı. Nokta, yolun hücre içinde
# en uzun olduğu hücreye (o parçanın ortasına) konur. Bulunamazsa uyarı verilir.
ZORUNLU_YOLLAR = (
    "Süleyman Seba Caddesi",    # stadyumun yanı (eski adı Spor Caddesi, Akaretler)
    "Ihlamurdere Caddesi",      # Beşiktaş çarşı → Ihlamur
    "Şair Nedim Caddesi",       # Beşiktaş çarşı içi
    "Serencebey Yokuşu",        # Barbaros → Yıldız (OSM'de residential)
    "Ortabahçe Caddesi",        # Beşiktaş çarşı içi
    "Balmumcu Deresi Sokağı",   # Balmumcu
)


def build_query(cells: list, include_minor: bool) -> str:
    """Sadece verilen hücrelerin içindeki yolları ister (hücre başına bir kutu)."""
    classes = MAIN_CLASSES + ("|" + MINOR_CLASSES if include_minor else "")
    parts = []
    for c in cells:
        s, n, w, e = decode_bbox(c)
        parts.append(f'way["highway"~"^({classes})$"]({s:.5f},{w:.5f},{n:.5f},{e:.5f});')
    return "[out:json][timeout:120];(" + "".join(parts) + ");out geom;"


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
                    min_sep: float, minor_fallback: bool,
                    ek_max_rank: int = 99, ek_min_len: float = 0.0) -> list:
    """
    Önce ana yollar (rank <= tertiary), önem sırasına ve hücre içi uzunluğa göre.
    Hücrede hiç ana yol yoksa ve minor_fallback açıksa sokaklardan seçer.
    İlk 2 seçimden sonrakiler (3., 4. ...) ancak rank <= ek_max_rank ve hücre içi
    uzunluk >= ek_min_len ise alınır; böylece zayıf yollar için kota harcanmaz.
    """
    pool = [g for g in groups.values()
            if g["length"] >= min_len and g["rank"] <= MAIN_MAX_RANK]
    if not pool and minor_fallback:
        pool = [g for g in groups.values() if g["length"] >= min_len]
    pool.sort(key=lambda g: (g["rank"], -g["length"]))

    chosen = []
    for g in pool:
        if len(chosen) >= 2 and (g["rank"] > ek_max_rank or g["length"] < ek_min_len):
            continue
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

def get_osm(cache_path: Path, cells: list, include_minor: bool, refresh: bool) -> dict:
    """Önbellek bu hücreleri kapsıyorsa onu kullanır, yoksa OSM'den indirir."""
    if cache_path.exists() and not refresh:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        meta = data.get("_meta", {})
        if "cells" in meta:
            kapsiyor = set(cells) <= set(meta["cells"])
        else:   # eski önbellek: tek büyük kutu
            bb = [decode_bbox(c) for c in cells]
            s, n = min(b[0] for b in bb), max(b[1] for b in bb)
            w, e = min(b[2] for b in bb), max(b[3] for b in bb)
            mb = meta.get("bbox", [1e9, -1e9, 1e9, -1e9])
            kapsiyor = mb[0] <= s and mb[1] >= n and mb[2] <= w and mb[3] >= e
        if kapsiyor and (meta.get("minor") or not include_minor):
            print(f"  Önbellek kullanılıyor: {cache_path}  (yenilemek için --refresh)")
            return data
        print("  Önbellek bu istek için yetersiz, OSM'den yeniden indiriliyor...")

    print(f"  OSM (Overpass) sorgulanıyor ({len(cells)} hücre), 10-60 sn sürebilir...")
    data = fetch_overpass(build_query(cells, include_minor))
    data["_meta"] = {"cells": sorted(cells), "minor": include_minor}
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="OSM → TomTom sorgu noktası seçici")
    ap.add_argument("--hucreler", default=None,
                    help="Hücre kaynağı: .parquet (geohash sütunu) veya kontrol.py'nin "
                         "ibb_hucreler.csv'si; verilmezse eğitim verisi "
                         "(egitim/besiktas_trafik_lstm.parquet)")
    ap.add_argument("--haric", default=None,
                    help="Dışarıda bırakılacak hücreler, virgülle ayrılmış. Varsayılan: Asya "
                         "yakası listesi (HARIC_HUCRELER). Hiçbirini çıkarmamak için: --haric \"\"")
    ap.add_argument("--min-ay", type=int, default=12,
                    help="--hucreler CSV ise: en az kaç ayda görünen hücreler (varsayılan 12)")
    ap.add_argument("--per-cell", type=int, default=3,
                    help="Hücre başına en fazla kaç yol (varsayılan 3; rota önerisi için yol çeşitliliği)")
    ap.add_argument("--ucuncu-min-sinif", default="secondary",
                    choices=[k for k, v in sorted(CLASS_RANK.items(), key=lambda x: x[1])],
                    help="3. (ve sonraki) yol en az bu sınıfta olmalı (varsayılan secondary)")
    ap.add_argument("--ucuncu-min-uzunluk", type=float, default=200,
                    help="3. (ve sonraki) yol hücre içinde en az bu kadar metre olmalı (varsayılan 200)")
    ap.add_argument("--min-len", type=float, default=80,
                    help="Hücre içinde en az bu kadar metre yol olmalı (varsayılan 80)")
    ap.add_argument("--min-sep", type=float, default=100,
                    help="Aynı hücredeki iki nokta arası en az metre (varsayılan 100)")
    ap.add_argument("--max-points", type=int, default=100,
                    help="Toplam nokta üst sınırı (varsayılan 100: saatlik sorguda 2.400 istek/gün, "
                         "TomTom ücretsiz kotası ~2.500; 0 = sınır yok)")
    ap.add_argument("--bos-per-cell", type=int, default=1,
                    help="Hiç noktası olmayan hücreler için seçilecek nokta sayısı (varsayılan 1)")
    ap.add_argument("--bos-min-len", type=float, default=20,
                    help="Boş hücrelerde kabul edilen en kısa yol uzunluğu, metre (varsayılan 20)")
    ap.add_argument("--minor-fallback", action="store_true",
                    help="Kısa ana yol da yoksa, SADECE boş kalan hücrelere sokak ekle")
    ap.add_argument("--cache", default="osm_ham.json")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default="tomtom_noktalar",
                    help="Çıktı dosya öneki (.json ve .csv üretilir)")
    args = ap.parse_args()

    print("\n[1/3] İBB hücreleri belirleniyor...")
    haric = None if args.haric is None else [x.strip() for x in args.haric.split(",") if x.strip()]
    cells, hucre_geohash, kaynak = load_cells(args.hucreler, args.min_ay, haric)
    print(f"  Kaynak: {kaynak}")
    print(f"  {len(cells)} adet 6 karakterli hücre")

    bboxes = {c: decode_bbox(c) for c in cells}

    print("\n[2/3] OpenStreetMap yolları alınıyor...")
    # Sokaklar da indirilir (zorunlu yollardan bazıları ve --minor-fallback için).
    # Otomatik seçim yine sadece ana yollarla yapılır (aşağıdaki ways_secim);
    # sokaklar sadece zorunlu yollarda ve boş hücre yedeğinde kullanılır.
    indir_minor = args.minor_fallback or bool(ZORUNLU_YOLLAR)
    data = get_osm(Path(args.cache), cells, indir_minor, args.refresh)
    ways = [el for el in data.get("elements", []) if el.get("type") == "way"]
    ways_secim = [el for el in ways
                  if CLASS_RANK.get(el.get("tags", {}).get("highway"), 99) <= MAIN_MAX_RANK]
    print(f"  {len(ways):,} yol parçası ({len(ways_secim):,} tanesi otomatik seçimde)")

    print("\n[3/3] Her hücre için nokta seçiliyor...")
    ek_max_rank = CLASS_RANK[args.ucuncu_min_sinif]
    points, bos_hucre, yedekler = [], [], []

    def nokta_ekle(cell: str, k, c: dict, zorunlu: bool, yedek: str = "") -> None:
        points.append({
            "id": f"{cell}-{k}", "road": c["road"],
            "lat": c["lat"], "lon": c["lon"], "length_m": c["length_m"],
            # geohash = eğitim verisindeki hücre geohash'i (noktanın kendisinin değil);
            # TomTom ölçümleri bu etiketle kaydedilip hücre bazında ortalanacak
            "cell": cell, "geohash": hucre_geohash[cell],
            "highway": c["highway"],
            "secim": "zorunlu" if zorunlu else "otomatik",
            "yedek": yedek,   # boş hücre yedeğiyle eklendiyse nedeni, değilse ""
            # zorunlu noktalar _pick=0 → --max-points kesiminde en son çıkarılır
            "_pick": 0 if zorunlu else k, "_rank": c["_rank"], "_len": c["_len"],
        })
        if yedek:
            yedekler.append((cell, c["road"], c["highway"], c["length_m"], yedek))

    for cell in cells:
        groups = groups_for_cell(bboxes[cell], ways_secim)
        chosen = select_for_cell(groups, args.per_cell, args.min_len,
                                 args.min_sep, False,
                                 ek_max_rank, args.ucuncu_min_uzunluk)
        yedek = ""
        if not chosen:   # 1. yedek: daha kısa ana yolu da kabul et
            chosen = select_for_cell(groups, args.bos_per_cell, args.bos_min_len,
                                     args.min_sep, False)
            yedek = f"kısa ana yol (>= {args.bos_min_len:g} m)"
        if not chosen and args.minor_fallback:   # 2. yedek: sokak
            groups = groups_for_cell(bboxes[cell], ways)
            chosen = select_for_cell(groups, args.bos_per_cell, args.bos_min_len,
                                     args.min_sep, True)
            yedek = "sokak (ana yol yok)"
        if not chosen:
            bos_hucre.append(cell)
            continue
        for k, c in enumerate(chosen, 1):
            nokta_ekle(cell, k, c, zorunlu=False, yedek=yedek)

    # ── Zorunlu yollar ───────────────────────────────────────────────────────
    tum_gruplar = {cell: groups_for_cell(bboxes[cell], ways) for cell in cells}
    zorunlu_eklenen, zorunlu_zaten, zorunlu_yok = [], [], []
    for ad in ZORUNLU_YOLLAR:
        adaylar = [(cell, g[ad]) for cell, g in tum_gruplar.items() if ad in g]
        if not adaylar:
            zorunlu_yok.append(ad)
            continue
        cell, g = max(adaylar, key=lambda x: x[1]["length"])
        if any(p["cell"] == cell and p["road"] == ad for p in points):
            zorunlu_zaten.append(f"{ad} ({cell})")
            continue
        lat, lon = middle_point(g["best_pts"])
        k = sum(1 for p in points if p["cell"] == cell) + 1
        nokta_ekle(cell, k, {
            "road": ad, "highway": g["highway"],
            "lat": round(lat, 6), "lon": round(lon, 6),
            "length_m": int(round(min(g["length"], 500))),
            "_rank": g["rank"], "_len": g["length"],
        }, zorunlu=True)
        zorunlu_eklenen.append(f"{ad} ({cell})")
    if bos_hucre:   # zorunlu yol boş bir hücreye düştüyse artık boş değil
        bos_hucre =[c for c in bos_hucre if not any(p["cell"] == c for p in points)]

    if args.max_points and len(points) > args.max_points:
        points.sort(key=lambda p: (p["_pick"], p["_rank"], -p["_len"]))
        cikarilan = len(points) - args.max_points
        points = points[:args.max_points]
        print(f"  [BİLGİ] Kota sınırı için {cikarilan} nokta çıkarıldı "
              f"(ikinci/üçüncü seçimlerden, önemsiz yollar önce; zorunlular korunur)")
    points.sort(key=lambda p: (p["cell"], int(p["id"].rsplit("-", 1)[1])))

    temiz = [{k: v for k, v in p.items() if not k.startswith("_")} for p in points]
    Path(args.out + ".json").write_text(
        json.dumps(temiz, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["id", "cell", "geohash", "road", "highway", "secim", "lat", "lon", "length_m", "yedek"]
    with open(args.out + ".csv", "w", encoding="utf-8-sig") as f:
        f.write(",".join(cols) + "\n")
        for p in temiz:
            f.write(",".join('"{}"'.format(str(p[c]).replace('"', "'")) for c in cols) + "\n")

    # ── Özet ────────────────────────────────────────────────────────────────
    hucre_sayisi = len({p["cell"] for p in temiz})
    hucre_nokta = {}
    for p in temiz:
        hucre_nokta[p["cell"]] = hucre_nokta.get(p["cell"], 0) + 1
    nokta_dagilimi = {}
    for n in hucre_nokta.values():
        nokta_dagilimi[n] = nokta_dagilimi.get(n, 0) + 1
    sinif = {}
    for p in temiz:
        sinif[p["highway"]] = sinif.get(p["highway"], 0) + 1

    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  Nokta seçilen hücre : {hucre_sayisi} / {len(cells)}")
    print(f"  Toplam nokta        : {len(temiz)}")
    print(f"  Yol sınıfı dağılımı : {sinif}")
    for cell, road, hw, uz, yedek in yedekler:
        print(f"  Yedekle eklendi     : {cell} → {road} ({hw}, hücre içinde {uz} m) [{yedek}]")
    if bos_hucre:
        print(f"  Nokta bulunamayan   : {len(bos_hucre)} hücre → {', '.join(bos_hucre)}")
        print(f"     (deniz olabilir; sokak yedeği için --minor-fallback, "
              f"ya da JSON'a elle nokta ekleyin)")
    print(f"  Hücre başına nokta  : " + ", ".join(
        f"{n} nokta → {h} hücre" for n, h in sorted(nokta_dagilimi.items())))
    print(f"  Zorunlu yol eklendi : {len(zorunlu_eklenen)} / {len(ZORUNLU_YOLLAR)}"
          + (f" → {', '.join(zorunlu_eklenen)}" if zorunlu_eklenen else ""))
    if zorunlu_zaten:
        print(f"  Zaten seçiliydi     : {', '.join(zorunlu_zaten)}")
    if zorunlu_yok:
        print(f"  [UYARI] OSM'de bulunamayan zorunlu yol: {', '.join(zorunlu_yok)}")
    print(f"\n  Tahmini TomTom isteği:")
    for interval in (60, 10):
        print(f"    {interval:>2} dk aralıkla: {len(temiz) * (1440 // interval):>7,} istek/gün")
    print(f"\n  Çıktı: {args.out}.json  ve  {args.out}.csv")
    print(f"  Haritada görmek için CSV'yi Google My Maps'e yükleyebilirsiniz.")
    print(f"  © OpenStreetMap contributors")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
