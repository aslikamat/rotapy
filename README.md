## Veri şeması görselleştirmesi için dbdiagram.io bağıntısı : 
https://dbdiagram.io/d/traffic_schema-6a50ea1836d348d120b52b01

# Beşiktaş Dinamik Rota Planlama Sistemi

Beşiktaş ilçesinde kriz ve etkinlik anlarında toplu ulaşım yoğunluğunu öngören
ve alternatif güzergah öneren yapay zeka tabanlı sistem.

**TÜBİTAK 2209-A** kapsamında geliştirilmektedir.  
**Galatasaray Üniversitesi** — 2025-2026

---

## Proje Yapısı

```
besiktas-rota/
├── egitim/                         # Model eğitimi
│   ├── gecmis_trafik_ibb.py        # İBB 2020-2024 saatlik trafik verisi
│   ├── gecmis_etkinlik_scraper.py  # Radar TR + Ticketmaster + Mackolik
│   ├── gecmis_hava_durumu.py       # Open-Meteo geçmiş hava verisi
│   ├── birlestirme.py              # Veri kaynaklarını birleştir
│   └── lstm_egitim.py              # LSTM modeli eğit
│
└── sistem/                         # Canlı sistem
    ├── anlik_trafik_tomtom_alansal.py  # TomTom anlık trafik (58 segment)
    ├── anlik_etkinlik_scraper.py       # Radar TR anlık etkinlikler
    └── anlik_hava_durumu.py            # Open-Meteo anlık hava
```

---

## Kurulum

```bash
pip install -r requirements.txt
playwright install chromium
```

---

## Kullanım

### 1. Geçmiş veri topla

```bash
cd egitim

# İBB trafik verisi (2023-2024)
python gecmis_trafik_ibb.py --quick

# Etkinlik verisi
python gecmis_etkinlik_scraper.py --start 2023-01-01 --end 2024-12-31

# Hava durumu
python gecmis_hava_durumu.py --start 2023-01-01 --end 2024-12-31
```

### 2. Verileri birleştir

```bash
python birlestirme.py
# Çıktı: lstm_egitim_verisi.parquet + veri_kalite_raporu.json
```

### 3. Modeli eğit

```bash
# Önce test et (3 epoch)
python lstm_egitim.py --test

# Tam eğitim (30 epoch)
python lstm_egitim.py --epochs 30
# Çıktı: besiktas_lstm_model/
```

### 4. Anlık veri (canlı sistem)

```bash
cd sistem
python anlik_trafik_tomtom_alansal.py --key TOMTOM_KEY --interval 10
python anlik_etkinlik_scraper.py --check
python anlik_hava_durumu.py --mod anlik
```

---

## API Anahtarları

| Servis | Nereden alınır | Ücretsiz mi? |
|---|---|---|
| TomTom Traffic | developer.tomtom.com | ✅ 2500 istek/gün |
| Ticketmaster | developer.ticketmaster.com | ✅ |
| Setlist.fm | setlist.fm/settings/api | ✅ |

---

## Veri Kaynakları

| Kaynak | Veri | Dönem |
|---|---|---|
| İBB Açık Veri Portalı | Saatlik trafik yoğunluğu | 2020-2025 |
| Radar Türkiye | 16 platform birleşik etkinlik | 2020-günümüz |
| Open-Meteo | Saatlik hava durumu | 1940-günümüz |
| TomTom Flow API | Anlık segment bazlı trafik | Gerçek zamanlı |

## Kaynak Site,Kategoriler : 
Biletix,"Konser, Spor, Tiyatro, Festival, Aile, Sanat, Stand-up, Eğitim"
Passo,"Spor (Futbol, Basketbol, Voleybol), Konser, Tiyatro, Sergi, Seminer, Kulüp"
Biletinial,"Sinema, Tiyatro, Konser, Spor, Çocuk, Seminer, Sergi"
Mobilet,"Konser, Tiyatro, Festival, Atölye, Spor, Stand-up, Sergi"
Bubilet,"Konser, Tiyatro, Festival, Eğitim, Stand-up, Çocuk"
Biletino,"Parti, Festival, Zirve, Eğitim, Kamp, Elektronik Müzik, Konferans"
İticket,"Konser, Tiyatro, Sergi, Müze, Spor, Çocuk Etkinlikleri"
İBB Şehir Tiyatroları,"Yetişkin Tiyatrosu, Çocuk Tiyatrosu, Müzikal, Çağdaş Gösteri"
Tiyatrolar.com.tr,"Tiyatro Oyunları, Stand-up, Söyleşi, Okuma Tiyatrosu"
Zorlu PSM,"Konser, Müzikal, Tiyatro, Festival, Kulüp Etkinlikleri, Atölye"
DasDas,"Tiyatro Oyunları, Konser, Stand-up, Çocuk Oyunları, Performans Sanatları"
Maximum Uniq,"Açıkhava Konserleri, Tiyatro, Stand-up, Festival, Kurumsal Etkinlikler"
Biletwise,"Futbol, Basketbol, Voleybol, Konser, Tiyatro, Özel Etkinlikler"
Eventbrite,"Seminer, Atölye, Zirve, Networking, Sergi, Ücretsiz Etkinlikler, Teknoloji Toplantıları"
Moda Sahnesi,"Tiyatro, Bağımsız Sinema, Atölye, Seminer, Konser"
Bilet.com (Etkinlik),"Tema Park, Müze, Konser, Tiyatro, Spor Müsabakaları"

---

## Yöntem

**LSTM** (Long Short-Term Memory) sinir ağı ile 6 saatlik geçmiş veriye bakarak
30-60 dakika sonrasının trafik yoğunluğunu tahmin eder.

Tahminler **NetworkX** graf modeli üzerinde **Dijkstra algoritması** ile
işlenerek en az yoğun rotayı belirler.

Etkinlik etkisi **mesafe bazlı yarıçap** ile hesaplanır:
```
yarıçap (km) = 0.5 × √(kapasite / 500) (?)
```

---

## Gereksinimler

- Python 3.12+
- macOS / Linux / Windows
- İnternet bağlantısı (veri çekme için)


----

## Beşiktaş bölgesi için geohashler : 
['sxk9s3', 'sxk9s5', 'sxk9s6', 'sxk9s2', 'sxk9s8', 'sxk9s0', 'sxk9kk', 'sxk9se', 'sxk9sk', 'sxk9kt', 'sxk9kr', 'sxk9e9', 'sxk9s9', 'sxk9sh', 'sxk97s', 'sxk9s4', 'sxk9ec', 'sxk9ks', 'sxk9ef', 'sxk9kx', 'sxk9eb', 'sxk9km', 'sxk9ed', 'sxk9s7', 'sxk9eu', 'sxk9kp', 'sxk97t', 'sxk9s1', 'sxk97y', 'sxk9sd', 'sxk97w', 'sxk9ee', 'sxk9es', 'sxk9ss', 'sxk97z', 'sxk9kh', 'sxk9e8', 'sxk9kw', 'sxk97x', 'sxk9eg', 'sxk9kq', 'sxk9kj', 'sxk9kn', 'sxk97v', 'sxk97u']



