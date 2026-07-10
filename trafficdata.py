import requests
import pandas as pd
import pygeohash as pgh
from datetime import datetime
import time

# TomTom API Anahtarın
API_KEY = "9dqoh1Ho4b8xTvYB3tnZCtGEb5qUS2S1"

# Beşiktaş bölgesini temsil eden Geohash listesi (Düğümler)
BESIKTAS_GEOHASHES = ["sxk9s3", "sxk9s5", "sxk9s6", "sxk9s2", "sxk9s8", "sxk9s0", "sxk9kk", "sxk9se", "sxk9sk", "sxk9kt", "sxk9kr", "sxk9e9", "sxk9s9", "sxk9sh", "sxk97s", "sxk9s4", "sxk9ec", "sxk9ks", "sxk9ef", "sxk9kx", "sxk9eb", "sxk9km", "sxk9ed", "sxk9s7", "sxk9eu", "sxk9kp", "sxk97t", "sxk9s1", "sxk97y", "sxk9sd", "sxk97w", "sxk9ee", "sxk9es", "sxk9ss", "sxk97z", "sxk9kh", "sxk9e8", "sxk9kw", "sxk97x", "sxk9eg", "sxk9kq", "sxk9kj", "sxk9kn", "sxk97v", "sxk97u"]
def get_traffic_label(ratio):
    """Yoğunluk oranına göre sözel yorum satırı belirler."""
    if ratio >= 0.90:
        return "Düşük Yoğunluk (Açık Yol)"
    elif ratio >= 0.70:
        return "Orta Yoğunluk (Akıcı Trafik)"
    elif ratio >= 0.40:
        return "Yüksek Yoğunluk (Yoğun Trafik)"
    else:
        return "Çok Yüksek Yoğunluk (Kilit Trafik)"
    
def fetch_instant_traffic(geohash_list, api_key):
    traffic_data_list = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for g_hash in geohash_list:
        # Geohash'i koordinata çevir
        lat, lon = pgh.decode(g_hash)
        
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            flow_data = data.get('flowSegmentData', {})
            
            current_speed = flow_data.get('currentSpeed')
            free_flow_speed = flow_data.get('freeFlowSpeed')
            
            # 0'a bölünme hatasını önlemek için kontrol
            free_flow_speed = free_flow_speed if free_flow_speed else 1
            congestion_ratio = current_speed / free_flow_speed if current_speed is not None else 1

            # Sözel yorum satırını hesapla
            traffic_label = get_traffic_label(congestion_ratio)

            traffic_data_list.append({
                "timestamp": current_time,
                "geohash": g_hash,
                "current_speed_kmh": current_speed,
                "free_flow_speed_kmh": free_flow_speed,
                "congestion_ratio": round(congestion_ratio, 3),
                "traffic_condition": traffic_label  # Yeni eklenen sözel yorum sütunu
            
            })
        else:
            print(f"Hata! Geohash: {g_hash} - Kodu: {response.status_code}")
            
        # API sınırlarına takılmamak için her istekte yarım saniye bekle
        time.sleep(0.5)
        
    return pd.DataFrame(traffic_data_list)

# Sadece bir kere çalışır ve anlık veriyi getirir
df_traffic = fetch_instant_traffic(BESIKTAS_GEOHASHES, API_KEY)

# Terminalde tabloyu görmeye devam etmek istersen bu satır kalabilir
print(df_traffic)

# Çıktıyı CSV dosyası olarak kaydet
df_traffic.to_csv("besiktas_anlik_trafik.csv", index=False)
